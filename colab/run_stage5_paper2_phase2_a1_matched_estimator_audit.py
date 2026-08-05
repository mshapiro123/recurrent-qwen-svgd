"""Stage, execute, and publish the read-only A1 matched-estimator audit."""

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
RUN_ID = "stage5_paper2_phase2_a1_matched_estimator_audit_20260805"
PRIOR_RUN_ID = "stage5_paper2_phase2_staged_a1_20260805"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_STAGE0A = DRIVE_ROOT / STAGE0A_ID / "private/stage0a"
DRIVE_PRIOR = DRIVE_ROOT / PRIOR_RUN_ID / "private/a1"
DRIVE_RUN = DRIVE_ROOT / RUN_ID
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)
PROTOCOL = ROOT / "training/paper2_phase2_staged_repilot_preregistration.json"
EXPECTED_CHECKPOINTS = {
    0: "9815592e5358fbde535bec27d102717f4f9fe4a0beb9f649f0d0879f88db2c58",
    1: "f3538465223c2f09f286bbb276631b3ce9e60a7c3ecd43bf677d4d4c4dfb6e4e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        print("a1_matched_audit_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a1_matched_audit_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_a1_matched_estimator_audit_status",
                "status": status,
                "updated_at_unix": time.time(),
                "gpu_name": os.environ.get("STAGE5_A1_AUDIT_GPU_NAME"),
                "optimizer_updates": 0,
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
    print(f"a1_matched_audit_status status={status} details={details}", flush=True)


def validate_prior_receipts() -> dict[str, str]:
    observed = {}
    for seed, expected in EXPECTED_CHECKPOINTS.items():
        arm = DRIVE_PRIOR / f"alpha_0p5_seed_{seed}"
        checkpoint = arm / "a1_resume.pt"
        calibration = arm / "a1_calibration.json"
        for path in (checkpoint, calibration):
            print(f"a1_matched_audit_preflight path={path} exists={path.exists()}", flush=True)
            if not path.is_file():
                raise FileNotFoundError(path)
        observed_sha = sha256_file(checkpoint)
        if observed_sha != expected:
            raise RuntimeError(
                f"seed {seed} prior checkpoint mismatch: expected {expected}, observed {observed_sha}"
            )
        observed[f"seed_{seed}_checkpoint_sha256"] = observed_sha
        observed[f"seed_{seed}_calibration_sha256"] = sha256_file(calibration)
    return observed


def stage_inputs() -> tuple[Path, Path, Path, Path]:
    scratch_root = Path("/content/local-scratch")
    if not scratch_root.is_dir():
        scratch_root = Path("/content")
    local = scratch_root / "recurrent-qwen-svgd-stage" / RUN_ID
    stage0a = local / "stage0a"
    canonicalizer = local / DRIVE_CANONICALIZER.name
    cache = local / "a1_matched_audit_cache.pt"
    prior = local / "prior_a1"
    stage0a.mkdir(parents=True, exist_ok=True)
    prior.mkdir(parents=True, exist_ok=True)
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
    for seed in EXPECTED_CHECKPOINTS:
        source_arm = DRIVE_PRIOR / f"alpha_0p5_seed_{seed}"
        destination_arm = prior / f"alpha_0p5_seed_{seed}"
        destination_arm.mkdir(parents=True, exist_ok=True)
        for name in ("a1_resume.pt", "a1_calibration.json"):
            shutil.copy2(source_arm / name, destination_arm / name)
    return stage0a, canonicalizer, cache, prior


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record A1 matched-estimator audit [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [STAGE0A_SUMMARY, DRIVE_STAGE0A / "sample_manifest.jsonl", DRIVE_CANONICALIZER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    prior_hashes = validate_prior_receipts()
    write_status("staging_inputs", prior_hashes=prior_hashes)
    stage0a, canonicalizer, cache, prior = stage_inputs()
    write_status("matched_estimator_audit")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_a1_matched_estimator_audit",
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--stage0a_private",
            str(stage0a),
            "--canonicalizer",
            str(canonicalizer),
            "--cache",
            str(cache),
            "--prior_private",
            str(prior),
            "--protocol",
            str(PROTOCOL),
            "--output_dir",
            str(RUN_DIR),
            "--device",
            "cuda",
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["optimizer_updates"] != 0 or summary["a2_launched"] is not False:
        raise RuntimeError("audit violated its read-only boundary")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN_DIR / "summary.json", receipt_dir / "summary.json")
    write_status("publishing", decision=summary["decision"])
    commit = publish()
    write_status("complete", decision=summary["decision"], publish_commit=commit)
    print(json.dumps({"summary": summary, "publish_commit": commit}, indent=2, sort_keys=True))
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
