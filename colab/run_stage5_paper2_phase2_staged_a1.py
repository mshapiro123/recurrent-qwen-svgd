"""Stage, run, and publish the locked Phase-2 staged A1 experiment."""

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
RUN_ID = "stage5_paper2_phase2_staged_a1_20260805"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_STAGE0A = DRIVE_ROOT / STAGE0A_ID / "private/stage0a"
DRIVE_RUN = DRIVE_ROOT / RUN_ID
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)
PROTOCOL = ROOT / "training/paper2_phase2_staged_repilot_preregistration.json"
CONSTANTS = ROOT / "training/paper2_phase2_dc2_constants.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_file(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_staged_a1_status",
                "status": status,
                "updated_at_unix": time.time(),
                "gpu_name": os.environ.get("STAGE5_STAGED_A1_GPU_NAME"),
                "gpu_vram_gib": os.environ.get("STAGE5_STAGED_A1_GPU_VRAM_GIB"),
                "a2_launched": False,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"staged_a1_status status={status} details={details}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=500)
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
    if returncode:
        print("staged_a1_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("staged_a1_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def validate_locked_inputs() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    observed = {
        "constants_lf_sha256": sha256_lf_file(CONSTANTS),
        "canonicalizer_sha256": sha256_file(DRIVE_CANONICALIZER),
    }
    print(f"staged_a1_locked_inputs observed={json.dumps(observed, sort_keys=True)}", flush=True)
    if observed["constants_lf_sha256"] != protocol["constants"]["lf_sha256"]:
        raise RuntimeError("V1d constants LF hash does not match the staged registration")
    if observed["canonicalizer_sha256"] != protocol["canonicalizer"]["sha256"]:
        raise RuntimeError("canonicalizer hash does not match the staged registration")


def stage_inputs() -> tuple[Path, Path, Path]:
    scratch_root = Path("/content/local-scratch")
    if not scratch_root.is_dir():
        scratch_root = Path("/content")
    local = scratch_root / "recurrent-qwen-svgd-stage" / RUN_ID
    stage0a = local / "stage0a"
    canonicalizer = local / DRIVE_CANONICALIZER.name
    cache = local / "staged_a1_cache.pt"
    stage0a.mkdir(parents=True, exist_ok=True)
    for relative in (
        "sample_manifest.jsonl",
        "lattice",
        "model_cache/student_0p5b",
        "model_cache/teacher_14b",
    ):
        source = DRIVE_STAGE0A / relative
        destination = stage0a / relative
        print(f"staged_a1_stage source={source} destination={destination}", flush=True)
        if source.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            run(["rsync", "-a", "--info=progress2", f"{source}/", f"{destination}/"])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(DRIVE_CANONICALIZER, canonicalizer)
    return stage0a, canonicalizer, cache


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 staged A1 results [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [STAGE0A_SUMMARY, DRIVE_STAGE0A / "sample_manifest.jsonl", DRIVE_CANONICALIZER]
    for path in required:
        print(f"staged_a1_preflight path={path} exists={path.exists()}", flush=True)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing staged-A1 inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    validate_locked_inputs()
    write_status("staging_inputs")
    stage0a, canonicalizer, cache = stage_inputs()
    write_status("a1_calibration_and_training")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "training.run_paper2_phase2_staged_a1",
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--stage0a_private",
            str(stage0a),
            "--canonicalizer",
            str(canonicalizer),
            "--cache",
            str(cache),
            "--output_dir",
            str(RUN_DIR),
            "--private_dir",
            str(DRIVE_RUN / "private/a1"),
            "--device",
            "cuda",
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary.get("a2_launched") is not False:
        raise RuntimeError("A1 launcher violated the strategy review boundary")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.glob("*.json"):
        shutil.copy2(path, receipt_dir / path.name)
    write_status(
        "publishing",
        run_status=summary["status"],
        verdicts=[arm["verdict"] for arm in summary["arms"]],
    )
    commit = publish()
    write_status(
        "complete",
        publish_commit=commit,
        run_status=summary["status"],
        verdicts=[arm["verdict"] for arm in summary["arms"]],
        strategy_review_required_before_a2=True,
    )
    print(json.dumps({"summary": summary, "publish_commit": commit}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete_with_strategy_gate_required" else 2


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
