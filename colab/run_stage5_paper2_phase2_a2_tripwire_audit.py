"""Stage, run, and publish the read-only A2 gradient-tripwire audit."""

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
RUN_ID = "stage5_paper2_phase2_a2_tripwire_audit_20260806"
SOURCE_RUN_ID = "stage5_paper2_phase2_a2_resume_20260805"
A1_RUN_ID = "stage5_paper2_phase2_staged_a1_resume_20260805"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
CALIBRATION_ID = "stage5_paper2_phase2_a2_calibration_20260805"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
SOURCE_SUMMARY = ROOT / "outputs/stage5" / SOURCE_RUN_ID / "summary.json"
CALIBRATION_SUMMARY = ROOT / "outputs/stage5" / CALIBRATION_ID / "summary.json"
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
PROTOCOL = ROOT / "training/paper2_phase2_staged_repilot_preregistration.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_STAGE0A = DRIVE_ROOT / STAGE0A_ID / "private/stage0a"
DRIVE_A1 = DRIVE_ROOT / A1_RUN_ID / "private/a1"
DRIVE_SOURCE = DRIVE_ROOT / SOURCE_RUN_ID / "private/a2"
DRIVE_RUN = DRIVE_ROOT / RUN_ID
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
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
    if returncode:
        print("a2_tripwire_audit_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a2_tripwire_audit_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_a2_tripwire_audit_status",
                "status": status,
                "updated_at_unix": time.time(),
                "gpu_name": os.environ.get("STAGE5_PHASE2_A2_TRIPWIRE_AUDIT_GPU_NAME"),
                "optimizer_updates_persisted": 0,
                "training_authorized": False,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"a2_tripwire_audit_status status={status} details={details}", flush=True)


def _complete_staged_root(path: Path) -> bool:
    required = [
        path / "stage0a/sample_manifest.jsonl",
        path / "stage0a/lattice",
        path / "stage0a/model_cache/student_0p5b/summary.json",
        path / "stage0a/model_cache/teacher_14b/summary.json",
        path / "learned_mixture_rrr_seed_20260814.pt",
        path / "a2_cache.pt",
        path / "a1_seed_0_step_1000.pt",
        path / "a1_seed_1_step_1000.pt",
    ]
    return all(item.exists() for item in required)


def stage_inputs() -> tuple[Path, Path, Path, dict[int, Path], dict[int, Path]]:
    existing_candidates = [
        Path("/content/recurrent-qwen-svgd-stage") / SOURCE_RUN_ID,
        Path("/content/local-scratch/recurrent-qwen-svgd-stage") / SOURCE_RUN_ID,
    ]
    existing = next((path for path in existing_candidates if _complete_staged_root(path)), None)
    if existing is not None:
        print(f"a2_tripwire_reuse_staged_inputs={existing}", flush=True)
        stage0a = existing / "stage0a"
        canonicalizer = existing / DRIVE_CANONICALIZER.name
        cache = existing / "a2_cache.pt"
        a1 = {seed: existing / f"a1_seed_{seed}_step_1000.pt" for seed in (0, 1)}
    else:
        scratch = Path("/content/local-scratch")
        if not scratch.is_dir():
            scratch = Path("/content")
        local = scratch / "recurrent-qwen-svgd-stage" / RUN_ID
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
        a1 = {}
        for seed in (0, 1):
            destination = local / f"a1_seed_{seed}_step_1000.pt"
            shutil.copy2(DRIVE_A1 / f"alpha_0p5_seed_{seed}/a1_resume_amended.pt", destination)
            a1[seed] = destination
    public = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    expected_resume = {
        int(arm["seed"]): arm["checkpoint"]["sha256"]
        for arm in public["arms"]
        if arm["arm"] == "full_a2"
    }
    runtime = stage0a.parent / "tripwire_checkpoints"
    runtime.mkdir(parents=True, exist_ok=True)
    resume = {}
    for seed in (0, 1):
        destination = runtime / f"seed_{seed}_full_step_237.pt"
        source = DRIVE_SOURCE / f"seed_{seed}_full_a2/resume.pt"
        if not destination.is_file() or sha256_file(destination) != expected_resume[seed]:
            shutil.copy2(source, destination)
        if sha256_file(destination) != expected_resume[seed]:
            raise RuntimeError(f"seed {seed} staged tripwire checkpoint SHA mismatch")
        resume[seed] = destination
    return stage0a, canonicalizer, cache, a1, resume


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record A2 gradient-tripwire audit [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [SOURCE_SUMMARY, CALIBRATION_SUMMARY, STAGE0A_SUMMARY, DRIVE_CANONICALIZER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing A2 tripwire audit inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("staging_inputs")
    stage0a, canonicalizer, cache, a1, resume = stage_inputs()
    write_status("read_only_gradient_and_counterfactual_audit")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_a2_tripwire_audit",
            "--protocol",
            str(PROTOCOL),
            "--a2_summary",
            str(SOURCE_SUMMARY),
            "--calibration_summary",
            str(CALIBRATION_SUMMARY),
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--stage0a_private",
            str(stage0a),
            "--canonicalizer",
            str(canonicalizer),
            "--cache",
            str(cache),
            "--a1_checkpoint_seed_0",
            str(a1[0]),
            "--a1_checkpoint_seed_1",
            str(a1[1]),
            "--resume_checkpoint_seed_0",
            str(resume[0]),
            "--resume_checkpoint_seed_1",
            str(resume[1]),
            "--output_dir",
            str(RUN_DIR),
            "--device",
            "cuda",
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["optimizer_updates_persisted"] != 0 or summary["training_authorized"] is not False:
        raise RuntimeError("A2 tripwire audit crossed its read-only boundary")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN_DIR / "summary.json", receipt_dir / "summary.json")
    write_status("publishing")
    commit = publish()
    write_status("complete", publish_commit=commit)
    print(json.dumps({"publish_commit": commit, "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
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

