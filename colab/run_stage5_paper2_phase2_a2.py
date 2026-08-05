"""Stage, resume, evaluate, and publish the locked Phase-2 A2 matrix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESUME_MODE = os.environ.get("STAGE5_PHASE2_A2_RESUME_MODE", "0") == "1"
RUN_ID = (
    "stage5_paper2_phase2_a2_resume_20260805"
    if RESUME_MODE
    else "stage5_paper2_phase2_a2_20260805"
)
SOURCE_A2_RUN_ID = "stage5_paper2_phase2_a2_20260805"
A1_RUN_ID = "stage5_paper2_phase2_staged_a1_resume_20260805"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_STAGE0A = DRIVE_ROOT / STAGE0A_ID / "private/stage0a"
DRIVE_A1 = DRIVE_ROOT / A1_RUN_ID / "private/a1"
DRIVE_RUN = DRIVE_ROOT / RUN_ID
DRIVE_SOURCE_A2 = DRIVE_ROOT / SOURCE_A2_RUN_ID / "private/a2"
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)
PROTOCOL = ROOT / "training/paper2_phase2_staged_repilot_preregistration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=600)
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
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode not in allowed:
        print("a2_matrix_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a2_matrix_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)
    return returncode


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_a2_matrix_status",
                "status": status,
                "updated_at_unix": time.time(),
                "gpu_name": os.environ.get("STAGE5_PHASE2_A2_GPU_NAME"),
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"a2_matrix_status status={status} details={details}", flush=True)


def stage_static_inputs() -> tuple[Path, Path, Path, dict[int, Path]]:
    scratch_root = Path("/content/local-scratch")
    if not scratch_root.is_dir():
        scratch_root = Path("/content")
    local = scratch_root / "recurrent-qwen-svgd-stage" / RUN_ID
    stage0a = local / "stage0a"
    canonicalizer = local / DRIVE_CANONICALIZER.name
    cache = local / "a2_cache.pt"
    stage0a.mkdir(parents=True, exist_ok=True)
    for relative in (
        "sample_manifest.jsonl",
        "lattice",
        "model_cache/student_0p5b",
        "model_cache/teacher_14b",
    ):
        source = DRIVE_STAGE0A / relative
        destination = stage0a / relative
        if source.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            run(["rsync", "-a", "--info=progress2", f"{source}/", f"{destination}/"])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(DRIVE_CANONICALIZER, canonicalizer)
    checkpoints = {}
    for seed in (0, 1):
        source = DRIVE_A1 / f"alpha_0p5_seed_{seed}/a1_resume_amended.pt"
        destination = local / f"a1_seed_{seed}_step_1000.pt"
        shutil.copy2(source, destination)
        checkpoints[seed] = destination
    return stage0a, canonicalizer, cache, checkpoints


def stage_resume_checkpoints(lock: dict[str, object]) -> None:
    if not RESUME_MODE:
        return
    destination_root = DRIVE_RUN / "private/a2"
    expected_by_arm = lock["source_resume_sha256_by_arm"]
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            name = f"seed_{seed}_{arm}"
            source = DRIVE_SOURCE_A2 / name / "resume.pt"
            destination = destination_root / name / "resume.pt"
            if destination.is_file():
                print(f"a2_resume_destination_exists arm={name}", flush=True)
                continue
            if not source.is_file():
                raise FileNotFoundError(f"missing registered A2 resume source: {source}")
            expected = expected_by_arm[name]
            if sha256_file(source) != expected:
                raise RuntimeError(f"registered A2 resume source SHA mismatch for {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != expected:
                raise RuntimeError(f"staged A2 resume source SHA mismatch for {name}")
            print(f"a2_resume_staged arm={name} sha256={expected}", flush=True)


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        message = (
            "Record Phase 2 A2 resumed matrix [skip ci]"
            if RESUME_MODE
            else "Record Phase 2 A2 matrix [skip ci]"
        )
        run(["git", "commit", "-m", message])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    registration = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = registration["a2_lock_amendment_20260805"]
    if lock["status"] != "locked_before_a2_training":
        raise RuntimeError("A2 matrix is not locked")
    required = [STAGE0A_SUMMARY, DRIVE_CANONICALIZER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing A2 matrix inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("staging_inputs")
    stage0a, canonicalizer, cache, checkpoints = stage_static_inputs()
    if RESUME_MODE:
        resume_lock = registration["a2_step200_resume_amendment_20260805"]
        if resume_lock["status"] != "locked_before_a2_resumed_training":
            raise RuntimeError("A2 step-200 resume is not locked")
        stage_resume_checkpoints(resume_lock)
    for seed, checkpoint in checkpoints.items():
        expected = lock["a1_checkpoint_sha256_by_seed"][str(seed)]
        if sha256_file(checkpoint) != expected:
            raise RuntimeError(f"seed {seed} staged A1 checkpoint SHA mismatch")
    write_status("training_four_run_matrix")
    command = [
            sys.executable,
            "-u",
            "-m",
            "training.run_paper2_phase2_a2",
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--stage0a_private",
            str(stage0a),
            "--canonicalizer",
            str(canonicalizer),
            "--cache",
            str(cache),
            "--a1_checkpoint_seed_0",
            str(checkpoints[0]),
            "--a1_checkpoint_seed_1",
            str(checkpoints[1]),
            "--output_dir",
            str(RUN_DIR),
            "--private_dir",
            str(DRIVE_RUN / "private/a2"),
            "--device",
            "cuda",
        ]
    if RESUME_MODE:
        command.append("--resume_from_step200")
    returncode = run(
        command,
        allowed=(0, 2),
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.glob("*.json"):
        shutil.copy2(path, receipt_dir / path.name)
    write_status("publishing", child_returncode=returncode, matrix_status=summary["status"])
    commit = publish()
    write_status(
        "complete",
        publish_commit=commit,
        child_returncode=returncode,
        matrix_status=summary["status"],
        pairs=[{"seed": row["seed"], "verdict": row["verdict"]} for row in summary["pairs"]],
    )
    print(json.dumps({"publish_commit": commit, "summary": summary}, indent=2, sort_keys=True))
    return returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0, 2):
            try:
                write_status(
                    "failed",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    traceback=traceback.format_exc(),
                )
            except Exception as status_error:
                print(f"status_write_failed={status_error!r}", flush=True)
        raise
