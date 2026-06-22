"""Collect capability-ladder trace responses into a gated curriculum shard.

This CPU-only runner starts from either a ``stage5_capability_ladder_trace_jobs``
summary plus a response JSONL, or directly from a
``stage5_capability_ladder_trace_responses`` summary produced by
``training/run_curriculum_job_responses.py``. It verifies final answers, builds
traced scored rows, exports a capability-ladder curriculum without
``--allow_answer_only``, runs the SFT gate, backs up to Drive, and optionally
pushes safe summaries.
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


RUN_ID = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_RUN_ID") or time.strftime(
    "stage5_capability_ladder_trace_collection_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
WORK_DIR = ROOT / os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_WORK_DIR", f"data/curriculum/{RUN_ID}")
SOURCE_SUMMARY = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_SOURCE_SUMMARY", "").strip()
RESPONSES_JSONL = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_JSONL", "").strip()
BASE_KEY = os.environ.get("STAGE5_CAPABILITY_LADDER_BASE_KEY", "qwen_0_5b")
MID_KEY = os.environ.get("STAGE5_CAPABILITY_LADDER_MID_KEY", "qwen_1_5b")
HIGH_KEYS = os.environ.get("STAGE5_CAPABILITY_LADDER_HIGH_KEYS", "qwen_3b")
HIGH_TARGET_LOOP = os.environ.get("STAGE5_CAPABILITY_LADDER_HIGH_TARGET_LOOP", "3")
MIN_POSITIVE_ROWS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MIN_POSITIVE_ROWS", "1")
MIN_MODE_ROWS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MIN_MODE_ROWS", "")
PUSH_RESULTS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BACKUP_DRIVE = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_BACKUP_DRIVE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
REFUSE_GPU_RUNTIME = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_REFUSE_GPU", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_GPU_RUNTIME_FOR_CPU_WORK = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU",
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
            "Refusing to collect capability-ladder trace responses on an attached GPU runtime: "
            + "; ".join(gpus)
            + ". Switch to CPU runtime, or set STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU=1 deliberately."
        )
    if gpus:
        print("GPU attached but CPU-work override is enabled:", gpus, flush=True)
    else:
        print("No GPU attached; good for trace-response collection.", flush=True)


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
            "Missing source summary. Set STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_SOURCE_SUMMARY "
            "or update config/stage5_current_source_summary.txt."
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def trace_jobs_summary_for_collection(source_summary: Path) -> Path:
    payload = read_json(source_summary)
    if payload.get("kind") != "stage5_capability_ladder_trace_responses":
        return source_summary
    raw = str(payload.get("source_summary") or "").strip()
    if not raw:
        raise ValueError(f"{source_summary} is a trace-response summary but has no source_summary.")
    trace_jobs_summary = resolve_path(raw)
    if not trace_jobs_summary.exists():
        raise FileNotFoundError(
            f"Trace-response summary points at missing trace-job summary: {trace_jobs_summary}"
        )
    return trace_jobs_summary


def response_path_from_response_summary(source_summary: Path) -> Path | None:
    payload = read_json(source_summary)
    if payload.get("kind") != "stage5_capability_ladder_trace_responses":
        return None
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    raw = str(artifacts.get("responses_jsonl") or "").strip()
    if not raw:
        raise ValueError(f"{source_summary} is a trace-response summary but has no artifacts.responses_jsonl.")
    return resolve_path(raw)


def artifact_path(payload: dict[str, Any], name: str) -> Path:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    raw = str(artifacts.get(name) or "").strip()
    if not raw:
        raise ValueError(f"Source summary does not contain artifacts.{name}.")
    return resolve_path(raw)


def report_scored_path(source_summary: Path) -> Path:
    payload = read_json(source_summary)
    report_path = artifact_path(payload, "report_json")
    report = read_json(report_path)
    source = report.get("source") if isinstance(report.get("source"), dict) else {}
    raw = str(source.get("scored_jsonl") or "").strip()
    if not raw:
        raise ValueError(f"{report_path} does not contain source.scored_jsonl.")
    return resolve_path(raw)


def drive_search_roots() -> list[Path]:
    drive = Path("/content/drive/MyDrive")
    if not drive.exists():
        return []
    return [
        drive / "recurrent-qwen-svgd" / "stage5_capability_ladder_trace_responses",
        drive / "recurrent-qwen-svgd" / "stage5_capability_ladder_trace_jobs",
        drive / "recurrent-qwen-svgd" / "stage5_capability_ladder",
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
    candidates = sorted(
        candidates,
        key=lambda item: (run_id_hint not in str(item), len(str(item))),
    )
    if not candidates:
        raise FileNotFoundError(f"Missing {path} and no Drive backup {filename} was found.")
    source = candidates[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    return {"restored": True, "path": path_for_cli(path), "source": str(source), "candidates": len(candidates)}


def resolve_responses(source_summary: Path) -> Path:
    if RESPONSES_JSONL:
        path = resolve_path(RESPONSES_JSONL)
        if path.exists():
            return path
        run_id_hint = path.parent.name or str(read_json(source_summary).get("run_id") or source_summary.parent.name)
        restored = restore_if_missing(path, filename=path.name, run_id_hint=run_id_hint)
        print(f"restored_responses={restored}", flush=True)
        if path.exists():
            return path
        return path
    local_candidates = [
        source_summary.parent / "trace_responses.jsonl",
        source_summary.parent / "capability_ladder_trace_responses.jsonl",
        source_summary.parent / "responses.jsonl",
    ]
    for path in local_candidates:
        if path.exists():
            return path
    run_id_hint = str(read_json(source_summary).get("run_id") or source_summary.parent.name)
    drive_candidates: list[Path] = []
    for root in drive_search_roots():
        if root.exists():
            for filename in ("trace_responses.jsonl", "capability_ladder_trace_responses.jsonl", "responses.jsonl"):
                drive_candidates.extend(root.rglob(filename))
    drive_candidates = sorted(
        drive_candidates,
        key=lambda item: (run_id_hint not in str(item), len(str(item))),
    )
    if not drive_candidates:
        raise FileNotFoundError(
            "No trace response JSONL found. Set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_JSONL "
            "or place trace_responses.jsonl next to the trace-job summary."
        )
    dest = RUN_DIR / "trace_responses.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_candidates[0], dest)
    return dest


def resolve_response_summary_responses(response_summary: Path) -> Path:
    responses = response_path_from_response_summary(response_summary)
    if responses is None:
        raise ValueError(f"{response_summary} is not a trace-response summary.")
    if responses.exists():
        return responses
    payload = read_json(response_summary)
    run_id_hint = responses.parent.name or str(payload.get("run_id") or response_summary.parent.name)
    restored = restore_if_missing(responses, filename=responses.name, run_id_hint=run_id_hint)
    print(f"restored_responses={restored}", flush=True)
    return responses


def collect_traces(scored_jsonl: Path, jobs_jsonl: Path, responses_jsonl: Path) -> tuple[Path, Path, dict[str, Any]]:
    output_jsonl = RUN_DIR / "scored_rows_with_traces.jsonl"
    report_json = RUN_DIR / "trace_collection_report.json"
    run(
        [
            sys.executable,
            "training/collect_capability_ladder_trace_outputs.py",
            "--scored_jsonl",
            path_for_cli(scored_jsonl),
            "--jobs_jsonl",
            path_for_cli(jobs_jsonl),
            "--responses_jsonl",
            path_for_cli(responses_jsonl),
            "--output_jsonl",
            path_for_cli(output_jsonl),
            "--report_json",
            path_for_cli(report_json),
        ],
        log_name="collect_capability_ladder_trace_outputs.log",
    )
    return output_jsonl, report_json, read_json(report_json)


def build_curriculum(traced_jsonl: Path) -> Path:
    run(
        [
            sys.executable,
            "training/build_capability_ladder_curriculum.py",
            "--input_jsonl",
            path_for_cli(traced_jsonl),
            "--work_dir",
            path_for_cli(WORK_DIR),
            "--base_key",
            BASE_KEY,
            "--mid_key",
            MID_KEY,
            "--high_keys",
            HIGH_KEYS,
            "--high_target_loop",
            HIGH_TARGET_LOOP,
        ],
        log_name="build_capability_ladder_curriculum.log",
    )
    return WORK_DIR / "summary.json"


def gate_curriculum(curriculum_summary: Path) -> Path:
    output = RUN_DIR / "curriculum_sft_gate.json"
    output_md = RUN_DIR / "curriculum_sft_gate.md"
    cmd = [
        sys.executable,
        "training/check_curriculum_sft_gate.py",
        "--work_dir",
        path_for_cli(WORK_DIR),
        "--summary_json",
        path_for_cli(curriculum_summary),
        "--output_json",
        path_for_cli(output),
        "--output_md",
        path_for_cli(output_md),
        "--min_positive_rows",
        MIN_POSITIVE_ROWS,
        "--max_loop_target",
        str(max(int(HIGH_TARGET_LOOP), 3)),
    ]
    if MIN_MODE_ROWS:
        cmd.extend(["--min_mode_rows", MIN_MODE_ROWS])
    run(cmd, check=False, log_name="check_curriculum_sft_gate.log")
    return output


def backup_to_drive(paths: list[Path]) -> dict[str, Any]:
    if not BACKUP_DRIVE:
        return {"enabled": False}
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        return {"enabled": True, "available": False}
    dest_root = drive_root / "recurrent-qwen-svgd" / "stage5_capability_ladder_trace_collection" / RUN_ID
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
        print("No safe trace-collection result changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", "Record capability-ladder trace collection"], check=True, log_name="git_commit.log")
    run(["git", "push", "origin", "main"], check=True, log_name="git_push.log")


def write_summary(
    *,
    source_summary: Path,
    response_summary: Path | None = None,
    scored_jsonl: Path,
    jobs_jsonl: Path,
    responses_jsonl: Path,
    traced_jsonl: Path,
    collection_report: Path,
    collection_payload: dict[str, Any],
    curriculum_summary: Path,
    gate_json: Path,
    restore_report: dict[str, Any],
    drive_backup: dict[str, Any],
) -> Path:
    gate_payload = read_json(gate_json) if gate_json.exists() else {}
    curriculum_payload = read_json(curriculum_summary)
    go = gate_payload.get("go") is True
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_capability_ladder_trace_collection",
        "status": "trace_curriculum_gate_ready" if go else "trace_curriculum_needs_review",
        "source_summary": path_for_cli(source_summary),
        "restore": restore_report,
        "collection": {
            "accepted_rows": int(collection_payload.get("accepted_rows") or 0),
            "status_counts": collection_payload.get("status_counts", {}),
            "target_loop_counts": collection_payload.get("target_loop_counts", {}),
            "tier_counts": collection_payload.get("tier_counts", {}),
        },
        "curriculum": {
            "summary_json": path_for_cli(curriculum_summary),
            "work_dir": path_for_cli(WORK_DIR),
            "counts": curriculum_payload.get("counts", {}),
        },
        "gate": gate_payload,
        "artifacts": {
            "scored_jsonl": path_for_cli(scored_jsonl),
            "jobs_jsonl": path_for_cli(jobs_jsonl),
            "responses_jsonl": path_for_cli(responses_jsonl),
            "traced_scored_rows": path_for_cli(traced_jsonl),
            "collection_report": path_for_cli(collection_report),
            "curriculum_summary": path_for_cli(curriculum_summary),
            "curriculum_gate": path_for_cli(gate_json),
            "run_dir": path_for_cli(RUN_DIR),
        },
        "drive_backup": drive_backup,
        "next_action": (
            "If the gate is green, run the recurrent SFT gate/training path on the traced capability-ladder curriculum."
            if go
            else "Inspect trace collection and gate output before any recurrent SFT."
        ),
    }
    if response_summary is not None:
        payload["response_summary"] = path_for_cli(response_summary)
    summary = RUN_DIR / "summary.json"
    write_json(summary, payload)
    (RUN_DIR / "summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return summary


def markdown_summary(payload: dict[str, Any]) -> str:
    collection = payload["collection"]
    curriculum = payload["curriculum"]
    return "\n".join(
        [
            f"# Capability-Ladder Trace Collection: {payload['run_id']}",
            "",
            f"- Status: `{payload['status']}`",
            f"- Accepted traced rows: `{collection['accepted_rows']}`",
            f"- Collection status counts: `{collection['status_counts']}`",
            f"- Target loops: `{collection['target_loop_counts']}`",
            f"- Curriculum rows: `{curriculum.get('counts', {}).get('positive_sft_rows', 0)}`",
            f"- Gate go: `{payload.get('gate', {}).get('go')}`",
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
    collection_source_summary = trace_jobs_summary_for_collection(source_summary)
    response_summary = source_summary if source_summary != collection_source_summary else None
    source_payload = read_json(collection_source_summary)
    run_id_hint = str(source_payload.get("run_id") or collection_source_summary.parent.name)
    jobs_jsonl = artifact_path(source_payload, "jobs_jsonl")
    report_json = artifact_path(source_payload, "report_json")
    restore_report = {
        "jobs": restore_if_missing(jobs_jsonl, filename=jobs_jsonl.name, run_id_hint=run_id_hint),
        "report": restore_if_missing(report_json, filename=report_json.name, run_id_hint=run_id_hint),
    }
    scored_jsonl = report_scored_path(collection_source_summary)
    restore_report["scored"] = restore_if_missing(scored_jsonl, filename=scored_jsonl.name, run_id_hint=run_id_hint)
    responses_jsonl = (
        resolve_responses(collection_source_summary)
        if RESPONSES_JSONL or response_summary is None
        else resolve_response_summary_responses(response_summary)
    )
    traced_jsonl, collection_report, collection_payload = collect_traces(scored_jsonl, jobs_jsonl, responses_jsonl)
    curriculum_summary = build_curriculum(traced_jsonl)
    gate_json = gate_curriculum(curriculum_summary)
    drive_backup = backup_to_drive([RUN_DIR, WORK_DIR, traced_jsonl, collection_report, curriculum_summary, gate_json])
    summary = write_summary(
        source_summary=collection_source_summary,
        response_summary=response_summary,
        scored_jsonl=scored_jsonl,
        jobs_jsonl=jobs_jsonl,
        responses_jsonl=responses_jsonl,
        traced_jsonl=traced_jsonl,
        collection_report=collection_report,
        collection_payload=collection_payload,
        curriculum_summary=curriculum_summary,
        gate_json=gate_json,
        restore_report=restore_report,
        drive_backup=drive_backup,
    )
    print(f"summary_json={path_for_cli(summary)}", flush=True)
    print(f"status={read_json(summary)['status']}", flush=True)
    safe_commit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
