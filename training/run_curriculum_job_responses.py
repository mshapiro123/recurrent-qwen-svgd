"""Run curriculum prompt jobs through a provider-neutral response backend.

The default backend is a no-cost dry run. The ``command`` backend invokes an
external process per job, passes the job JSON on stdin, and records stdout as
``response_text``. This keeps provider-specific API code outside the curriculum
verification pipeline while standardizing response JSONL output.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
        rows.append(row)
    return rows


def existing_job_ids(path: str | Path) -> set[str]:
    out = Path(path)
    if not out.exists():
        return set()
    seen: set[str] = set()
    for row in read_jsonl(out):
        job_id = str(row.get("job_id") or "")
        if job_id:
            seen.add(job_id)
    return seen


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def truncate_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"...[truncated {len(value) - limit} chars]"


def response_id(job: dict[str, Any], *, prefix: str) -> str:
    return f"{prefix}-{job.get('job_id', 'unknown')}"


def dry_run_response(job: dict[str, Any], *, include_prompt: bool = False) -> dict[str, Any]:
    prompt = str(job.get("prompt") or "")
    response_text = prompt if include_prompt else f"DRY RUN: {job.get('stage')} {job.get('job_id')}"
    return {
        "job_id": job.get("job_id"),
        "response_id": response_id(job, prefix="dry-run"),
        "model": job.get("model"),
        "stage": job.get("stage"),
        "backend": "dry_run",
        "status": "ok",
        "response_text": response_text,
        "elapsed_sec": 0.0,
    }


def command_response(
    job: dict[str, Any],
    *,
    command: str,
    timeout_sec: float,
    stderr_limit: int,
) -> dict[str, Any]:
    argv = shlex.split(command, posix=(sys.platform != "win32"))
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(job),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        elapsed = time.monotonic() - started
        status = "ok" if proc.returncode == 0 else "error"
        return {
            "job_id": job.get("job_id"),
            "response_id": response_id(job, prefix="command"),
            "model": job.get("model"),
            "stage": job.get("stage"),
            "backend": "command",
            "status": status,
            "returncode": proc.returncode,
            "response_text": proc.stdout.strip(),
            "stderr": truncate_text(proc.stderr.strip(), limit=stderr_limit),
            "elapsed_sec": round(elapsed, 6),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        return {
            "job_id": job.get("job_id"),
            "response_id": response_id(job, prefix="command"),
            "model": job.get("model"),
            "stage": job.get("stage"),
            "backend": "command",
            "status": "timeout",
            "returncode": None,
            "response_text": "",
            "stderr": truncate_text(str(exc), limit=stderr_limit),
            "elapsed_sec": round(elapsed, 6),
        }


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    output_jsonl: str | Path,
    backend: str,
    command: str | None = None,
    limit: int | None = None,
    resume: bool = False,
    fail_fast: bool = False,
    timeout_sec: float = 120.0,
    sleep_sec: float = 0.0,
    include_prompt_in_dry_run: bool = False,
    stderr_limit: int = 4000,
) -> dict[str, Any]:
    completed = existing_job_ids(output_jsonl) if resume else set()
    selected = jobs[:limit] if limit is not None else jobs
    counts = {"written": 0, "skipped": 0, "errors": 0, "timeouts": 0}

    for job in selected:
        job_id = str(job.get("job_id") or "")
        if resume and job_id in completed:
            counts["skipped"] += 1
            continue
        if backend == "dry_run":
            row = dry_run_response(job, include_prompt=include_prompt_in_dry_run)
        elif backend == "command":
            if not command:
                raise ValueError("--command is required for --backend command")
            row = command_response(job, command=command, timeout_sec=timeout_sec, stderr_limit=stderr_limit)
        else:
            raise ValueError(f"Unsupported backend {backend!r}")

        append_jsonl(output_jsonl, row)
        counts["written"] += 1
        if row.get("status") == "error":
            counts["errors"] += 1
        if row.get("status") == "timeout":
            counts["timeouts"] += 1
        if fail_fast and row.get("status") != "ok":
            break
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return {
        "backend": backend,
        "jobs": len(jobs),
        "selected": len(selected),
        **counts,
        "output_jsonl": str(output_jsonl),
    }


def write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--backend", choices=("dry_run", "command"), default="dry_run")
    parser.add_argument("--command", help="External command for backend=command. Job JSON is passed on stdin.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--timeout_sec", type=float, default=120.0)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--include_prompt_in_dry_run", action="store_true")
    parser.add_argument("--stderr_limit", type=int, default=4000)
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs_jsonl)
    report = run_jobs(
        jobs,
        output_jsonl=args.output_jsonl,
        backend=args.backend,
        command=args.command,
        limit=args.limit,
        resume=args.resume,
        fail_fast=args.fail_fast,
        timeout_sec=args.timeout_sec,
        sleep_sec=args.sleep_sec,
        include_prompt_in_dry_run=args.include_prompt_in_dry_run,
        stderr_limit=args.stderr_limit,
    )
    write_report(args.report_json, report)
    print(f"backend={report['backend']}")
    print(f"selected={report['selected']}")
    print(f"written={report['written']}")
    print(f"skipped={report['skipped']}")
    print(f"errors={report['errors']}")
    print(f"timeouts={report['timeouts']}")
    return 1 if args.fail_fast and (report["errors"] or report["timeouts"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
