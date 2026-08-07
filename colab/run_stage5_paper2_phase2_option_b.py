"""Stage, run, resume, and publish the locked Phase-2 Option B matrix."""

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
RUN_ID = "stage5_paper2_phase2_option_b_20260807"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
A1_ID = "stage5_paper2_phase2_staged_a1_resume_20260805"
A2_ID = "stage5_paper2_phase2_a2_step237_continuation_20260806"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_ROOT / RUN_ID
OLD_SUMMARY = ROOT / "outputs/stage5" / OLD_ID / "summary.json"
NEW_PUBLIC_SUMMARY = ROOT / "outputs/stage5" / NEW_ID / "summary.json"
DRIVE_OLD = DRIVE_ROOT / OLD_ID / "private/stage0a"
DRIVE_NEW_ROOT = DRIVE_ROOT / NEW_ID
DRIVE_NEW = DRIVE_NEW_ROOT / "private/full"
DRIVE_NEW_FULL_SUMMARY = DRIVE_NEW_ROOT / "receipts/full_cache_summary.json"
DRIVE_FIXED_NEW = DRIVE_NEW_ROOT / "private/fixed_new_train_subset.json"
DRIVE_A1 = DRIVE_ROOT / A1_ID / "private/a1"
DRIVE_A2 = DRIVE_ROOT / A2_ID / "private/a2"
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)
REGISTRATION = ROOT / "training/paper2_phase2_option_b_preregistration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=800)
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
    code = process.wait()
    if code not in allowed:
        print("option_b_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("option_b_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(code, command)
    return code


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_option_b_status",
                "status": status,
                "updated_at_unix": time.time(),
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"option_b_status status={status} details={details}", flush=True)


def select_scratch() -> Path:
    candidates = [Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")]
    minimum = int(os.environ.get("STAGE5_OPTION_B_MIN_SCRATCH_BYTES", str(120 * 2**30)))
    for path in candidates:
        if path.is_dir() and shutil.disk_usage(path).free >= minimum:
            print(
                f"option_b_scratch path={path} free_gib={shutil.disk_usage(path).free / 2**30:.1f}",
                flush=True,
            )
            return path
    observed = {str(path): shutil.disk_usage(path).free if path.is_dir() else None for path in candidates}
    raise RuntimeError(f"Option B needs at least {minimum / 2**30:.0f} GiB scratch: {observed}")


def sync_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"missing Option B source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        run(["rsync", "-a", "--info=progress2", f"{source}/", f"{destination}/"])
    else:
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)


def stage_inputs() -> dict[str, object]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    scratch = select_scratch() / "recurrent-qwen-svgd-stage" / RUN_ID
    old = scratch / "old"
    new = scratch / "new"
    for relative in (
        "sample_manifest.jsonl",
        "lattice",
        "model_cache/student_0p5b",
        "model_cache/teacher_14b",
    ):
        sync_tree(DRIVE_OLD / relative, old / relative)
        sync_tree(DRIVE_NEW / relative, new / relative)
    canonicalizer = scratch / DRIVE_CANONICALIZER.name
    sync_tree(DRIVE_CANONICALIZER, canonicalizer)
    fixed_new = scratch / "fixed_new_train_subset.json"
    sync_tree(DRIVE_FIXED_NEW, fixed_new)
    new_full_summary = scratch / "full_cache_summary.json"
    sync_tree(DRIVE_NEW_FULL_SUMMARY, new_full_summary)
    a1: dict[int, Path] = {}
    endpoints: dict[tuple[int, str], Path] = {}
    for seed in (0, 1):
        a1[seed] = scratch / f"a1_seed_{seed}_step_1000.pt"
        sync_tree(DRIVE_A1 / f"alpha_0p5_seed_{seed}/a1_resume_amended.pt", a1[seed])
        for arm in ("full_a2", "draft_only_control"):
            destination = scratch / f"endpoint_seed_{seed}_{arm}.pt"
            sync_tree(DRIVE_A2 / f"seed_{seed}_{arm}/resume.pt", destination)
            expected = registration["source_checkpoints"][f"seed_{seed}_{arm}"]
            if sha256_file(destination) != expected:
                raise RuntimeError(f"Option B source endpoint SHA mismatch: seed={seed} arm={arm}")
            endpoints[(seed, arm)] = destination
    return {
        "scratch": scratch,
        "old": old,
        "new": new,
        "canonicalizer": canonicalizer,
        "fixed_new": fixed_new,
        "new_full_summary": new_full_summary,
        "a1": a1,
        "endpoints": endpoints,
    }


def publish() -> str:
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.glob("*.json"):
        shutil.copy2(path, receipt_dir / path.name)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase 2 Option B matrix [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [OLD_SUMMARY, NEW_PUBLIC_SUMMARY, DRIVE_NEW_FULL_SUMMARY, DRIVE_FIXED_NEW]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Option B locked inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("staging_inputs")
    staged = stage_inputs()
    scratch = staged["scratch"]
    write_status("building_caches_then_training", scratch=str(scratch))
    command = [
        sys.executable,
        "-u",
        "-m",
        "training.run_paper2_phase2_option_b",
        "--old_summary",
        str(OLD_SUMMARY),
        "--old_private",
        str(staged["old"]),
        "--new_public_summary",
        str(NEW_PUBLIC_SUMMARY),
        "--new_full_summary",
        str(staged["new_full_summary"]),
        "--new_private",
        str(staged["new"]),
        "--fixed_new_subset",
        str(staged["fixed_new"]),
        "--canonicalizer",
        str(staged["canonicalizer"]),
        "--old_cache",
        str(Path(scratch) / "old_cache.pt"),
        "--new_cache",
        str(Path(scratch) / "new_cache.pt"),
        "--a1_checkpoint_seed_0",
        str(staged["a1"][0]),
        "--a1_checkpoint_seed_1",
        str(staged["a1"][1]),
    ]
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            command.extend(
                [f"--endpoint_seed_{seed}_{arm}", str(staged["endpoints"][(seed, arm)])]
            )
    command.extend(
        [
            "--output_dir",
            str(RUN_DIR),
            "--private_dir",
            str(DRIVE_RUN / "private/option_b"),
            "--device",
            "cuda",
        ]
    )
    code = run(command, allowed=(0, 2))
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    write_status("publishing", matrix_status=summary["status"], child_returncode=code)
    commit = publish()
    write_status(
        "complete",
        matrix_status=summary["status"],
        child_returncode=code,
        publish_commit=commit,
    )
    print(json.dumps({"publish_commit": commit, "status": summary["status"]}, indent=2))
    return code


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
                print(f"option_b_status_write_failed={status_error!r}", flush=True)
        raise
