"""Stage, execute, and publish the locked read-once E1 confirmation pass."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_e1_confirmation_20260808"
EVAL_D_ID = "stage5_paper2_phase2_e1_eval_d_20260808"
OPTION_B_ID = "stage5_paper2_phase2_option_b_20260807"
A1_ID = "stage5_paper2_phase2_staged_a1_resume_20260805"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
DRIVE_EVAL_D = DRIVE_STAGE5 / EVAL_D_ID
DRIVE_OPTION_B = DRIVE_STAGE5 / OPTION_B_ID
PUBLIC_EVAL_D = ROOT / "outputs/stage5" / EVAL_D_ID / "receipts"
OPTION_B_SUMMARY = ROOT / "outputs/stage5" / OPTION_B_ID / "summary.json"
A1_SUMMARY = ROOT / "outputs/stage5" / A1_ID / "summary.json"


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code not in allowed:
        print("e1_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("e1_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(code, command)
    return code


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def status(status_value: str, **details: Any) -> None:
    payload = {
        "kind": "paper2_phase2_e1_confirmation_status_v1",
        "status": status_value,
        "updated_at_unix": time.time(),
        **details,
    }
    write_json(DRIVE_RUN / "receipts/status.json", payload)
    print("e1_status", json.dumps(payload, sort_keys=True), flush=True)


def select_scratch() -> Path:
    candidates = [Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")]
    usable = []
    for path in candidates:
        if path.is_dir() and os.access(path, os.W_OK):
            free = shutil.disk_usage(path).free
            if free >= 20 * 2**30:
                usable.append(("scratch" in str(path), free, path))
    if not usable:
        raise RuntimeError("E1 requires at least 20 GiB free local scratch")
    usable.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = usable[0][2] / "recurrent-qwen-svgd-stage" / RUN_ID
    selected.mkdir(parents=True, exist_ok=True)
    print(
        f"e1_scratch path={selected} free_gib={shutil.disk_usage(selected).free / 2**30:.1f}",
        flush=True,
    )
    return selected


def copy_verified(source: Path, destination: Path, expected_sha: str | None = None) -> Path:
    from training.paper2_phase2_e1_confirmation import sha256_file

    if not source.is_file():
        raise FileNotFoundError(f"missing E1 input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)
    if expected_sha is not None and source_sha != expected_sha:
        raise RuntimeError(f"E1 source SHA mismatch: {source}")
    if not destination.is_file() or sha256_file(destination) != source_sha:
        shutil.copy2(source, destination)
    if sha256_file(destination) != source_sha:
        raise RuntimeError(f"E1 staged-copy SHA mismatch: {destination}")
    return destination


def resolve_head(model_dir: Path) -> tuple[Path, str]:
    summary_path = model_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing E1 model-cache summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = summary["lm_head"]
    source = Path(receipt["path"])
    if not source.is_file():
        source = model_dir / source.name
    return source, str(receipt["sha256"])


def resolve_a1_checkpoints(summary_path: Path) -> dict[int, tuple[Path, str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved: dict[int, tuple[Path, str]] = {}
    for arm in summary["arms"]:
        if float(arm["alpha"]) != 0.5:
            continue
        seed = int(arm["seed"])
        checkpoint = arm["checkpoint"]
        resolved[seed] = (Path(checkpoint["path"]), str(checkpoint["sha256"]))
    if set(resolved) != {0, 1}:
        raise RuntimeError(f"E1 A1 summary did not resolve seeds 0 and 1: {summary_path}")
    return resolved


def publish() -> str:
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    summary = RUN_DIR / "summary.json"
    shutil.copy2(summary, receipt_dir / summary.name)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", summary.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record read-once Phase 2 E1 confirmation [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    from training.paper2_phase2_e1_confirmation import sha256_file

    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    scratch = select_scratch()
    status("staging_unscored_inputs", scratch=str(scratch))
    registration = json.loads(
        (ROOT / "training/paper2_phase2_e1_confirmation_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    cache = copy_verified(
        DRIVE_EVAL_D / "private/e1_eval_d_option_b_cache.pt",
        scratch / "e1_eval_d_option_b_cache.pt",
        registration["evaluation"]["private_cache_sha256"],
    )
    model_root = DRIVE_EVAL_D / "private/e1_eval_d/model_cache"
    student_source, student_sha = resolve_head(model_root / "student_0p5b")
    teacher_source, teacher_sha = resolve_head(model_root / "teacher_14b")
    student_head = copy_verified(student_source, scratch / "student_lm_head.pt", student_sha)
    teacher_head = copy_verified(teacher_source, scratch / "teacher14_lm_head.pt", teacher_sha)
    a1 = {}
    for seed, (source, expected_sha) in resolve_a1_checkpoints(A1_SUMMARY).items():
        a1[seed] = copy_verified(
            source,
            scratch / f"a1_seed_{seed}.pt",
            expected_sha,
        )
    endpoints = {}
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            name = f"seed_{seed}_{arm}"
            endpoints[name] = copy_verified(
                DRIVE_OPTION_B / f"private/option_b/{name}/resume.pt",
                scratch / f"{name}.pt",
                registration["checkpoints"][name]["sha256"],
            )

    required_public = {
        "freeze_receipt": PUBLIC_EVAL_D / "e1_eval_d_freeze_summary.json",
        "readiness_receipt": PUBLIC_EVAL_D / "e1_readiness.json",
        "sparse_qc_receipt": PUBLIC_EVAL_D / "e1_sparse_support_qc.json",
        "endpoint_receipt": PUBLIC_EVAL_D / "e1_endpoint_lock_preparation.json",
        "option_b_summary": OPTION_B_SUMMARY,
    }
    missing = [str(path) for path in required_public.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E1 public lock inputs: {missing}")
    lease = DRIVE_RUN / "receipts/read_once_lease.json"
    private_dir = DRIVE_RUN / "private/e1_confirmation"
    output = RUN_DIR / "summary.json"
    command = [
        sys.executable,
        "-u",
        "-m",
        "eval.eval_paper2_phase2_e1_confirmation",
    ]
    for flag, path in required_public.items():
        command.extend([f"--{flag}", str(path)])
    command.extend(
        [
            "--cache",
            str(cache),
            "--student_head",
            str(student_head),
            "--teacher_head",
            str(teacher_head),
            "--a1_checkpoint_seed_0",
            str(a1[0]),
            "--a1_checkpoint_seed_1",
            str(a1[1]),
        ]
    )
    for name, path in endpoints.items():
        command.extend([f"--endpoint_{name}", str(path)])
    command.extend(
        [
            "--rms_cap",
            "0.5508932316303252",
            "--private_dir",
            str(private_dir),
            "--lease",
            str(lease),
            "--output",
            str(output),
            "--device",
            "cuda",
        ]
    )
    status(
        "launching_read_once_scoring",
        cache_sha256=sha256_file(cache),
        lease=str(lease),
        read_once_scoring_spent=False,
    )
    run(command)
    summary = json.loads(output.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("read_once_scoring_spent") is not True:
        raise RuntimeError("E1 child did not produce a complete spent receipt")
    status(
        "publishing",
        read_once_scoring_spent=True,
        scripted_verdict=summary["scripted_verdict"],
    )
    commit = publish()
    status(
        "complete",
        read_once_scoring_spent=True,
        scripted_verdict=summary["scripted_verdict"],
        publish_commit=commit,
        summary_sha256=sha256_file(output),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "scripted_verdict": summary["scripted_verdict"],
                "publish_commit": commit,
                "summary": str(output),
                "summary_sha256": sha256_file(output),
                "read_once_scoring_spent": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            try:
                status(
                    "failed",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    traceback=traceback.format_exc(),
                )
            except Exception as status_error:
                print(f"e1_status_write_failed={status_error!r}", flush=True)
        raise
