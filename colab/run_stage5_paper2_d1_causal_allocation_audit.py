"""Restore, run, and publish the read-only D0 causal allocation audit."""

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
from training.speculative_depth_d0_postlock import D0_RUN_ID, validate_cache_summary


LOCK_RUN = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
D0_RUN = ROOT / "outputs" / "stage5" / D0_RUN_ID
RUN_ID = os.environ.get(
    "STAGE5_PAPER2_D1_AUDIT_RUN_ID", "stage5_paper2_d1_causal_allocation_audit_20260727"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
D0_DRIVE = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)
AUDIT_DRIVE = Path(
    os.environ.get(
        "STAGE5_PAPER2_D1_AUDIT_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}",
    )
)
EMA_SHA256 = "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status_event(status: str, **details: Any) -> None:
    payload = {"kind": "paper2_d1_causal_allocation_audit_status", "status": status, **details}
    print("d1_audit_status:", json.dumps(payload, sort_keys=True), flush=True)
    try:
        write_json(AUDIT_DRIVE / "receipts/status.json", payload)
    except Exception as error:
        print(f"d1_audit_status_write_failed={error!r}", flush=True)


def restore_inputs() -> dict[str, Path]:
    manifest = read_json(LOCK_RUN / "data_manifest.json")
    restored: dict[str, Path] = {}
    destination_dir = D0_RUN / "private_inputs"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in ("label_train", "calibration", "evaluation"):
        receipt = manifest["artifacts"][name]
        source = Path(receipt["drive_path"])
        destination = destination_dir / f"{name}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"missing locked D0 partition in Drive: {source}")
        if sha256_file(source) != receipt["sha256"]:
            raise RuntimeError(f"locked D0 partition hash mismatch: {name}")
        if not destination.exists() or sha256_file(destination) != receipt["sha256"]:
            shutil.copy2(source, destination)
        restored[name] = destination
    return restored


def restore_checkpoint() -> Path:
    explicit = os.environ.get("STAGE5_PAPER2_D1_AUDIT_CHECKPOINT", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.extend(
        [
            D0_DRIVE / "private/training/d0_ema_step_4000.pt",
            D0_DRIVE / "private/train/d0_ema_step_4000.pt",
            D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
        ]
    )
    diagnostics = []
    for candidate in candidates:
        exists = candidate.exists()
        observed = sha256_file(candidate) if exists else None
        diagnostics.append({"path": str(candidate), "exists": exists, "sha256": observed})
        if observed == EMA_SHA256:
            destination = RUN_DIR / "private_runtime/d0_ema_step_4000.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != EMA_SHA256:
                shutil.copy2(candidate, destination)
            print("d1_audit_checkpoint_resolution:", json.dumps(diagnostics), flush=True)
            return destination
    raise RuntimeError(f"could not restore post-D0 EMA checkpoint: {diagnostics}")


def publish() -> str:
    paths = [
        RUN_DIR / "summary.json",
        RUN_DIR / "summary.md",
        RUN_DIR / "causal_allocation_audit.png",
        RUN_DIR / "causal_allocation_audit.svg",
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"audit output missing before publish: {path}")
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    run(["git", "add", "-f", "--", *relative])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Paper Two D0 causal allocation audit [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    if os.environ.get("STAGE5_PAPER2_D1_ALLOW_TRAINING", "0") != "0":
        raise RuntimeError("D1 audit target forbids training authorization")
    d0_summary = read_json(D0_RUN / "summary.json")
    if d0_summary.get("status") != "complete":
        raise RuntimeError("D1 audit requires the completed registered D0 result")
    if d0_summary.get("interpretation_band") != "not_recoverable_at_pilot_scale":
        raise RuntimeError("D1 audit received an unexpected D0 verdict")
    cache_summary_path = D0_RUN / "labeling/summary.json"
    cache_summary = read_json(cache_summary_path)
    validate_cache_summary(cache_summary)
    calibration_private = D0_DRIVE / "private/eval/trained_teacher_shift_rows.json"
    evaluation_private = D0_DRIVE / "private/eval/evaluation_rows.json"
    if not calibration_private.exists():
        raise FileNotFoundError(f"missing trained calibration private rows: {calibration_private}")
    trained_shift = read_json(D0_RUN / "eval/trained_teacher_shift_summary.json")
    if sha256_file(calibration_private) != trained_shift["private_rows_sha256"]:
        raise RuntimeError("trained calibration private-row hash mismatch")
    natural = read_json(D0_RUN / "eval/natural_summary.json")
    if not evaluation_private.exists():
        raise FileNotFoundError(f"missing banked D0 private evaluation rows: {evaluation_private}")
    if sha256_file(evaluation_private) != natural["private_rows_sha256"]:
        raise RuntimeError("banked D0 private evaluation-row hash mismatch")
    inputs = restore_inputs()
    checkpoint = restore_checkpoint()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DRIVE.mkdir(parents=True, exist_ok=True)
    status_event("started", checkpoint_sha256=EMA_SHA256, training_authorized=False)
    run(
        [
            sys.executable,
            "-u",
            "eval/eval_speculative_depth_d1_causal_allocation.py",
            "--evaluation_jsonl",
            str(inputs["evaluation"]),
            "--calibration_jsonl",
            str(inputs["calibration"]),
            "--calibration_private_rows",
            str(calibration_private),
            "--evaluation_reference_rows",
            str(evaluation_private),
            "--label_train_jsonl",
            str(inputs["label_train"]),
            "--teacher_cache_summary",
            str(cache_summary_path),
            "--floor_summary",
            str(D0_RUN / "floor/summary.json"),
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            EMA_SHA256,
            "--output_dir",
            str(RUN_DIR),
            "--private_cache_dir",
            str(AUDIT_DRIVE / "private"),
            "--sample_positions",
            "100000",
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_D1_AUDIT_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("STAGE5_PAPER2_D1_AUDIT_ATTN", "sdpa"),
        ]
    )
    summary = read_json(RUN_DIR / "summary.json")
    if summary.get("optimizer_steps") != 0 or summary.get("training_started") is not False:
        raise RuntimeError("D1 causal allocation audit violated its read-only contract")
    receipt_dir = AUDIT_DRIVE / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "summary.md", "causal_allocation_audit.png", "causal_allocation_audit.svg"):
        shutil.copy2(RUN_DIR / name, receipt_dir / name)
    commit = publish()
    status_event("complete", publish_commit=commit, summary=str(RUN_DIR / "summary.json"))
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        status_event("errored", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise
