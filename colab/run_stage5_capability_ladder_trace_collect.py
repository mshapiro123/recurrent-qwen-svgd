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
MODEL_LADDER = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MODEL_LADDER", "").strip()
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
TRACE_RESPONSE_KIND = "stage5_capability_ladder_trace_responses"
TRACE_JOBS_KIND = "stage5_capability_ladder_trace_jobs"
SUPPORTED_SOURCE_KINDS = {TRACE_RESPONSE_KIND, TRACE_JOBS_KIND}


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


def valid_source_summary_payload(payload: dict[str, Any]) -> bool:
    kind = payload.get("kind")
    status = str(payload.get("status") or "")
    if kind == TRACE_RESPONSE_KIND:
        return status == "responses_ready"
    if kind == TRACE_JOBS_KIND:
        return status in {"", "ready"}
    return False


def valid_source_summary_path(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return valid_source_summary_payload(read_json(path))
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def summary_candidates() -> list[Path]:
    roots = [ROOT / "outputs" / "stage5", *drive_search_roots()]
    seen: set[str] = set()
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("summary.json"):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def latest_supported_source_summary() -> Path | None:
    ranked: list[tuple[int, float, int, Path]] = []
    for path in summary_candidates():
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not valid_source_summary_payload(payload):
            continue
        kind = payload.get("kind")
        kind_rank = 0 if kind == TRACE_RESPONSE_KIND else 1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        ranked.append((kind_rank, -mtime, len(str(path)), path))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def source_summary_error(path: Path) -> str:
    if not path.exists():
        return f"missing source summary: {path}"
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return f"unreadable source summary {path}: {exc}"
    kind = payload.get("kind")
    status = payload.get("status")
    if kind not in SUPPORTED_SOURCE_KINDS:
        return f"unsupported source summary kind {kind!r} in {path}"
    return f"source summary {path} has unsupported status {status!r} for kind {kind!r}"


def resolve_source_summary() -> Path:
    raw = SOURCE_SUMMARY
    explicit = bool(raw)
    if not raw:
        pointer = current_source_summary_file()
        if pointer.exists():
            raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        fallback = latest_supported_source_summary()
        if fallback is not None:
            print(f"using_latest_trace_source_summary={path_for_cli(fallback)}", flush=True)
            return fallback
        raise FileNotFoundError(
            "Missing source summary. Set STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_SOURCE_SUMMARY, "
            "update config/stage5_current_source_summary.txt, or run the trace-response target first."
        )
    path = resolve_path(raw)
    if valid_source_summary_path(path):
        return path
    if explicit:
        raise ValueError(source_summary_error(path))
    fallback = latest_supported_source_summary()
    if fallback is None:
        raise ValueError(
            source_summary_error(path)
            + "; no local or Drive-backed trace-response/job summary was found."
        )
    print(
        "current_source_summary_unusable="
        f"{source_summary_error(path)}; using_latest_trace_source_summary={path_for_cli(fallback)}",
        flush=True,
    )
    return fallback


def restore_summary_if_missing(path: Path, *, run_id_hint: str, expected_kind: str) -> Path:
    if path.exists():
        return path
    candidates: list[Path] = []
    for candidate in summary_candidates():
        try:
            payload = read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("kind") != expected_kind:
            continue
        candidates.append(candidate)
    candidates = sorted(
        candidates,
        key=lambda item: (
            run_id_hint not in str(item) and run_id_hint != str(read_json(item).get("run_id") or ""),
            len(str(item)),
        ),
    )
    if not candidates:
        raise FileNotFoundError(
            f"Missing {path} and no {expected_kind} summary backup matching {run_id_hint!r} was found."
        )
    source = candidates[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    print(f"restored_summary={{'path': '{path_for_cli(path)}', 'source': '{source}'}}", flush=True)
    return path


def trace_jobs_summary_for_collection(source_summary: Path) -> Path:
    payload = read_json(source_summary)
    if payload.get("kind") != TRACE_RESPONSE_KIND:
        return source_summary
    raw = str(payload.get("source_summary") or "").strip()
    if not raw:
        raise ValueError(f"{source_summary} is a trace-response summary but has no source_summary.")
    trace_jobs_summary = resolve_path(raw)
    run_id_hint = trace_jobs_summary.parent.name or str(payload.get("run_id") or source_summary.parent.name)
    return restore_summary_if_missing(trace_jobs_summary, run_id_hint=run_id_hint, expected_kind=TRACE_JOBS_KIND)


def response_path_from_response_summary(source_summary: Path) -> Path | None:
    payload = read_json(source_summary)
    if payload.get("kind") != TRACE_RESPONSE_KIND:
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


def format_model_ladder(ladder: Any) -> str:
    if not isinstance(ladder, list):
        return ""
    parts: list[str] = []
    for entry in ladder:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "").strip()
        loop = entry.get("target_loop_count")
        if key and isinstance(loop, int):
            parts.append(f"{key}:{loop}")
    return ",".join(parts)


def model_ladder_from_collection_source(source_summary: Path) -> str:
    if MODEL_LADDER:
        return MODEL_LADDER
    payload = read_json(source_summary)
    report_path: Path | None = None
    try:
        report_path = artifact_path(payload, "report_json")
    except ValueError:
        report_path = None
    if report_path is not None and report_path.exists():
        report_ladder = format_model_ladder(read_json(report_path).get("model_ladder"))
        if report_ladder:
            return report_ladder
    summary_ladder = format_model_ladder(payload.get("model_ladder"))
    if summary_ladder:
        return summary_ladder
    trace_jobs = payload.get("trace_jobs") if isinstance(payload.get("trace_jobs"), dict) else {}
    nested_ladder = format_model_ladder(trace_jobs.get("model_ladder"))
    if nested_ladder:
        return nested_ladder
    raw_probe = str(payload.get("source_summary") or "").strip()
    if raw_probe:
        probe = resolve_path(raw_probe)
        if probe.exists():
            probe_payload = read_json(probe)
            ladder_keys = probe_payload.get("ladder_keys") if isinstance(probe_payload.get("ladder_keys"), dict) else {}
            probe_ladder = format_model_ladder(ladder_keys.get("model_ladder"))
            if probe_ladder:
                return probe_ladder
    return ""


def max_loop_for_ladder(model_ladder: str) -> int:
    loops = [int(HIGH_TARGET_LOOP), 3]
    for raw_item in model_ladder.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        separator = ":" if ":" in raw_item else "=" if "=" in raw_item else None
        if separator is None:
            continue
        _key, raw_loop = raw_item.split(separator, 1)
        try:
            loops.append(int(raw_loop.strip()))
        except ValueError:
            continue
    return max(loops)


def reconstruct_scored_rows_if_possible(source_summary: Path, expected: Path) -> dict[str, Any] | None:
    trace_payload = read_json(source_summary)
    raw_probe = str(trace_payload.get("source_summary") or "").strip()
    if not raw_probe:
        return None
    probe_summary = resolve_path(raw_probe)
    if not probe_summary.exists():
        return None
    probe_payload = read_json(probe_summary)
    arc = probe_payload.get("arc") if isinstance(probe_payload.get("arc"), dict) else {}
    score_summaries = (
        probe_payload.get("score_summaries")
        if isinstance(probe_payload.get("score_summaries"), dict)
        else {}
    )
    required_arc = ["config", "split", "limit", "seed"]
    if not arc or not score_summaries or any(key not in arc for key in required_arc):
        return None

    result_specs: list[tuple[str, Path]] = []
    for key, summary in sorted(score_summaries.items()):
        if not isinstance(summary, dict):
            continue
        raw_path = str(summary.get("path") or "").strip()
        if not raw_path:
            continue
        path = resolve_path(raw_path)
        if not path.exists():
            return None
        result_specs.append((str(key), path))
    if not result_specs:
        return None

    expected.parent.mkdir(parents=True, exist_ok=True)
    tasks_jsonl = expected.parent / "arc_mcq_reconstructed.jsonl"
    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            str(arc["config"]),
            "--split",
            str(arc["split"]),
            "--limit",
            str(int(arc["limit"])),
            "--seed",
            str(int(arc["seed"])),
            "--output_jsonl",
            path_for_cli(tasks_jsonl),
        ],
        log_name="reconstruct_prepare_arc_mcq.log",
    )
    cmd = [
        sys.executable,
        "training/merge_capability_score_rows.py",
        "--tasks_jsonl",
        path_for_cli(tasks_jsonl),
        "--output_jsonl",
        path_for_cli(expected),
        "--verified_by",
        "benchmark_ground_truth",
        "--assume_decontaminated",
        "--prediction_as_solution",
    ]
    for key, path in result_specs:
        cmd.extend(["--result", f"{key}={path_for_cli(path)}"])
    run(cmd, log_name="reconstruct_merge_capability_score_rows.log")
    return {
        "restored": True,
        "path": path_for_cli(expected),
        "source": "reconstructed_from_trace_job_source_summary",
        "probe_summary": path_for_cli(probe_summary),
        "tasks_jsonl": path_for_cli(tasks_jsonl),
        "result_keys": [key for key, _path in result_specs],
    }


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


def build_curriculum(traced_jsonl: Path, collection_source_summary: Path) -> tuple[Path, str]:
    model_ladder = model_ladder_from_collection_source(collection_source_summary)
    cmd = [
        sys.executable,
        "training/build_capability_ladder_curriculum.py",
        "--input_jsonl",
        path_for_cli(traced_jsonl),
        "--work_dir",
        path_for_cli(WORK_DIR),
    ]
    if model_ladder:
        cmd.extend(["--model_ladder", model_ladder])
    else:
        cmd.extend(
            [
                "--base_key",
                BASE_KEY,
                "--mid_key",
                MID_KEY,
                "--high_keys",
                HIGH_KEYS,
                "--high_target_loop",
                HIGH_TARGET_LOOP,
            ]
        )
    run(cmd, log_name="build_capability_ladder_curriculum.log")
    return WORK_DIR / "summary.json", model_ladder


def gate_curriculum(curriculum_summary: Path, *, model_ladder: str = "") -> Path:
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
        str(max_loop_for_ladder(model_ladder)),
        "--allow_answer_line_verification",
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
    run(["git", "add", "-f", path_for_cli(RUN_DIR), path_for_cli(WORK_DIR), path_for_cli(pointer)], check=False, log_name="git_add.log")
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No safe trace-collection result changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", "Record capability-ladder trace collection [skip ci]"], check=True, log_name="git_commit.log")
    push = run(["git", "push", "origin", "main"], check=False, log_name="git_push.log")
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "origin", "main"], check=True, log_name="git_pull_rebase.log")
    run(["git", "push", "origin", "main"], check=True, log_name="git_push_retry.log")


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
    model_ladder: str,
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
            "model_ladder": model_ladder,
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
    if scored_jsonl.exists():
        restore_report["scored"] = {"restored": False, "path": path_for_cli(scored_jsonl), "source": "local"}
    else:
        reconstructed = reconstruct_scored_rows_if_possible(collection_source_summary, scored_jsonl)
        if reconstructed is not None and scored_jsonl.exists():
            restore_report["scored"] = reconstructed
        else:
            restore_report["scored"] = restore_if_missing(scored_jsonl, filename=scored_jsonl.name, run_id_hint=run_id_hint)
    responses_jsonl = (
        resolve_responses(collection_source_summary)
        if RESPONSES_JSONL or response_summary is None
        else resolve_response_summary_responses(response_summary)
    )
    traced_jsonl, collection_report, collection_payload = collect_traces(scored_jsonl, jobs_jsonl, responses_jsonl)
    curriculum_summary, model_ladder = build_curriculum(traced_jsonl, collection_source_summary)
    gate_json = gate_curriculum(curriculum_summary, model_ladder=model_ladder)
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
        model_ladder=model_ladder,
        restore_report=restore_report,
        drive_backup=drive_backup,
    )
    print(f"summary_json={path_for_cli(summary)}", flush=True)
    print(f"status={read_json(summary)['status']}", flush=True)
    safe_commit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
