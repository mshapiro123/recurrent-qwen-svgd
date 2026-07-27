"""Restore the locked D0 inputs, run teacher caching, and publish aggregate receipts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import D0_LOCK_COMMIT, D0_RUN_ID
from training.speculative_depth_d0_spec import DRAFTER_CHECKPOINT_SHA256, validate_locked_d0


# Safety marker: labeling proper only no optimizer no training
LOCK_RUN = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
RUN_DIR = ROOT / "outputs" / "stage5" / D0_RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)
CHECKPOINT_ALIAS = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
    "stage5_paper2_t1_lite_r_20260725/checkpoints/t1_lite_r_raw_step_10500.pt"
)
CHECKPOINT_STAGE_STATE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
    "stage5_paper2_t1_lite_r_20260725/checkpoints/stage_states/"
    "t1_lite_r_step_10500_raw.pt"
)


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code not in allowed:
        raise subprocess.CalledProcessError(code, command)
    return code


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_event(status: str, **details: Any) -> None:
    payload = {"kind": "paper2_d0_teacher_cache_status", "status": status, **details}
    print("d0_teacher_cache_status:", json.dumps(payload, sort_keys=True), flush=True)
    try:
        write_json(DRIVE_ROOT / "receipts" / "teacher_cache_status.json", payload)
    except Exception as error:
        print(f"d0_teacher_cache_status_write_failed={error!r}", flush=True)


def resolve_checkpoint_source(
    candidates: list[Path], *, expected_sha256: str
) -> tuple[Path, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            diagnostics.append({"path": key, "status": "missing"})
            continue
        observed = sha256_file(candidate)
        if observed != expected_sha256:
            diagnostics.append(
                {"path": key, "status": "sha_mismatch", "observed_sha256": observed}
            )
            continue
        diagnostics.append({"path": key, "status": "matched", "sha256": observed})
        return candidate, diagnostics
    raise FileNotFoundError(
        "No SHA-identical locked D0 drafter checkpoint was found. "
        f"expected_sha256={expected_sha256} candidates={json.dumps(diagnostics, sort_keys=True)}"
    )


def assert_lock_is_immutable() -> tuple[dict[str, Any], dict[str, Any]]:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", D0_LOCK_COMMIT, "HEAD"], cwd=ROOT
    ).returncode:
        raise RuntimeError(f"D0 lock commit is not an ancestor of HEAD: {D0_LOCK_COMMIT}")
    prereg = read_json(LOCK_RUN / "preregistration.json")
    validate_locked_d0(prereg)
    lock_receipt = read_json(LOCK_RUN / "lock_receipt.json")
    if lock_receipt.get("optimizer_steps") != 0:
        raise RuntimeError("D0 lock receipt was mutated after registration")
    return prereg, read_json(LOCK_RUN / "data_manifest.json")


def restore_private_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime_manifest = json.loads(json.dumps(manifest))
    private_dir = RUN_DIR / "private_inputs"
    for name in ("label_train", "calibration", "evaluation", "in_era_contrast"):
        receipt = runtime_manifest["artifacts"][name]
        source = Path(receipt["drive_path"])
        destination = private_dir / f"{name}.jsonl"
        print(
            f"d0_private_input_preflight name={name} path={source} exists={source.exists()}",
            flush=True,
        )
        if not source.exists():
            raise FileNotFoundError(f"Locked D0 private input is missing from Drive: {source}")
        observed = sha256_file(source)
        print(
            f"d0_private_input_sha name={name} observed={observed} expected={receipt['sha256']}",
            flush=True,
        )
        if observed != receipt["sha256"]:
            raise RuntimeError(
                f"Locked D0 private input hash mismatch: {name}; "
                f"observed={observed} expected={receipt['sha256']}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != receipt["sha256"]:
            shutil.copy2(source, destination)
        receipt["local_restore_path"] = str(destination)
    runtime_path = DRIVE_ROOT / "private" / "runtime_manifest.json"
    write_json(runtime_path, runtime_manifest)
    return {"manifest": runtime_manifest, "path": runtime_path}


def restore_checkpoint() -> tuple[Path, list[dict[str, Any]]]:
    explicit = os.environ.get("STAGE5_PAPER2_D0_DRAFTER_CHECKPOINT", "").strip()
    candidates = ([Path(explicit)] if explicit else []) + [CHECKPOINT_ALIAS, CHECKPOINT_STAGE_STATE]
    source, diagnostics = resolve_checkpoint_source(
        candidates, expected_sha256=DRAFTER_CHECKPOINT_SHA256
    )
    print(
        "d0_checkpoint_resolution:", json.dumps(diagnostics, sort_keys=True), flush=True
    )
    destination = RUN_DIR / "restored" / "t1_lite_r_raw_step_10500.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != DRAFTER_CHECKPOINT_SHA256:
        shutil.copy2(source, destination)
    return destination, diagnostics


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    run(["git", "add", "-f", "--", *relative])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 teacher cache produced no aggregate receipt changes")
    run(["git", "commit", "-m", "Record Paper Two D0 teacher cache [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    status_event("preflight_started", run_id=D0_RUN_ID, drive_root=str(DRIVE_ROOT))
    prereg, manifest = assert_lock_is_immutable()
    status_event("lock_validated", lock_commit=D0_LOCK_COMMIT)
    if prereg.get("labeling_gpu_authorized") is not True:
        raise RuntimeError("D0 labeling is not authorized by the landed lock")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    restored = restore_private_inputs(manifest)
    status_event("private_inputs_restored", manifest=str(restored["path"]))
    checkpoint, checkpoint_resolution = restore_checkpoint()
    status_event(
        "checkpoint_restored",
        checkpoint=str(checkpoint),
        checkpoint_sha256=DRAFTER_CHECKPOINT_SHA256,
        resolution=checkpoint_resolution,
    )
    summary_path = RUN_DIR / "labeling" / "summary.json"
    status_event("teacher_cache_started", output_summary=str(summary_path))
    run(
        [
            sys.executable,
            "eval/cache_speculative_depth_d0_teachers.py",
            "--preregistration",
            str(LOCK_RUN / "preregistration.json"),
            "--manifest",
            str(restored["path"]),
            "--checkpoint",
            str(checkpoint),
            "--cache_root",
            str(DRIVE_ROOT / "private" / "teacher_cache"),
            "--output_summary",
            str(summary_path),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_D0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        ]
    )
    summary = read_json(summary_path)
    summary["lock_preregistration"] = str((LOCK_RUN / "preregistration.json").relative_to(ROOT))
    summary["private_cache_drive_root"] = str(DRIVE_ROOT / "private" / "teacher_cache")
    write_json(summary_path, summary)
    drive_summary = DRIVE_ROOT / "receipts" / "labeling_summary.json"
    drive_summary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, drive_summary)
    summary_md = RUN_DIR / "labeling" / "summary.md"
    summary_md.write_text(
        "# Paper Two D0 Teacher Cache\n\n"
        "- Status: complete\n"
        "- Qwen2.5-7B: label-train, calibration, evaluation, in-era contrast\n"
        "- Qwen2.5-14B: calibration only\n"
        "- Teacher reload after completed cache: no\n"
        "- Optimizer steps: 0\n"
        f"- Private cache: `{DRIVE_ROOT / 'private' / 'teacher_cache'}`\n",
        encoding="utf-8",
    )
    publish_commit = publish([summary_path, summary_md])
    status_event("complete", publish_commit=publish_commit, summary=str(summary_path))
    print(json.dumps({**summary, "publish_commit": publish_commit}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        status_event(
            "errored",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
