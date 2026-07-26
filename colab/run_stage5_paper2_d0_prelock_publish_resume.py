"""Publish the completed D0 pre-lock receipts without rerunning model inference."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_paper2_d0_prelock import assert_model_and_dataset_revisions
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import (
    GOVERNING_DOCUMENT,
    GOVERNING_DOCUMENT_HANDOFF_SHA256,
    GOVERNING_DOCUMENT_SHA256,
    STACK_DATASET,
    STACK_REVISION,
    locked_d0_from_manifest,
    prelock_contract,
    validate_locked_d0,
)


RUN_ID = os.environ.get("STAGE5_PAPER2_D0_PRELOCK_RUN_ID", "stage5_paper2_d0_preregistration_20260726")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}",
    )
)
RECEIPT_PATHS = (
    Path("data_manifest.json"),
    Path("density/summary.json"),
    Path("source_access.json"),
    Path("lock_receipt.json"),
    Path("summary.json"),
    Path("summary.md"),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    process = subprocess.run(command, cwd=ROOT)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def restore_receipts_from_drive() -> None:
    for relative in RECEIPT_PATHS:
        source = DRIVE_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(
                f"Completed D0 receipt is missing from Drive: {source}. "
                "Do not rerun model inference until the runtime and Drive are checked."
            )
        destination = RUN_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)


def verify_private_data_backups(manifest: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, receipt in (manifest.get("artifacts") or {}).items():
        drive_path = Path(str(receipt.get("drive_path") or ""))
        expected = str(receipt.get("sha256") or "")
        if not drive_path.exists():
            raise FileNotFoundError(f"Frozen D0 artifact is missing from Drive: {name}: {drive_path}")
        observed = sha256_file(drive_path)
        if observed != expected:
            raise RuntimeError(f"Frozen D0 Drive artifact hash mismatch: {name}: {observed} != {expected}")
        verified[name] = {
            "drive_path": str(drive_path),
            "sha256": observed,
            "tokens": int(receipt.get("tokens", 0)),
        }
    return verified


def copy_updated_receipt_to_drive(path: Path) -> None:
    relative = path.relative_to(RUN_DIR)
    destination = DRIVE_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    if sha256_file(path) != sha256_file(destination):
        raise RuntimeError(f"Updated D0 receipt failed Drive verification: {relative}")


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    output_paths = [
        RUN_DIR.relative_to(ROOT) / "preregistration.json",
        RUN_DIR.relative_to(ROOT) / "data_manifest.json",
        RUN_DIR.relative_to(ROOT) / "density" / "summary.json",
        RUN_DIR.relative_to(ROOT) / "source_access.json",
        RUN_DIR.relative_to(ROOT) / "lock_receipt.json",
        RUN_DIR.relative_to(ROOT) / "summary.json",
        RUN_DIR.relative_to(ROOT) / "summary.md",
    ]
    run(["git", "add", "-f", "--", *[path.as_posix() for path in output_paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 publish recovery found no receipt changes to commit")
    run(["git", "commit", "-m", "Lock Paper Two D0 preregistration [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    governing = ROOT / GOVERNING_DOCUMENT
    if sha256_file(governing) != GOVERNING_DOCUMENT_SHA256:
        raise RuntimeError("D0 Draft 7 governing-document hash mismatch during recovery")
    restore_receipts_from_drive()
    manifest_path = RUN_DIR / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_artifacts = verify_private_data_backups(manifest)
    revisions = assert_model_and_dataset_revisions()
    contract = prelock_contract()
    manifest["dataset_revisions"] = revisions["revisions"]
    manifest["code_corpus"] = {
        "dataset": STACK_DATASET,
        "revision": STACK_REVISION,
        "lineage": "Stack_v1",
        "provenance_period": "in_pretraining_era",
        "content_store": "huggingface_direct_text",
        "terms_contract": contract["corpus"]["stack_smol"]["terms_contract"],
    }
    manifest["private_drive_artifacts_verified"] = verified_artifacts
    write_json(manifest_path, manifest)

    preregistration = locked_d0_from_manifest(manifest)
    validate_locked_d0(preregistration)
    preregistration_path = RUN_DIR / "preregistration.json"
    write_json(preregistration_path, preregistration)

    source_access_path = RUN_DIR / "source_access.json"
    source_access = json.loads(source_access_path.read_text(encoding="utf-8"))
    source_access.update(
        {
            "raw_source_executed": False,
            "storage_scope": "private_drive_token_ids_with_per_file_provenance",
            "stack_main_equals_pinned_at_recovery": True,
        }
    )
    write_json(source_access_path, source_access)

    lock_receipt_path = RUN_DIR / "lock_receipt.json"
    lock_receipt = json.loads(lock_receipt_path.read_text(encoding="utf-8"))
    lock_receipt.update(
        {
            "status": "recovered_ready_to_commit_locked_before_labeling",
            "governing_document": GOVERNING_DOCUMENT,
            "governing_document_sha256": GOVERNING_DOCUMENT_SHA256,
            "governing_document_handoff_sha256": GOVERNING_DOCUMENT_HANDOFF_SHA256,
            "hf_revisions": revisions,
            "private_drive_artifact_count_verified": len(verified_artifacts),
            "teacher_labeling_proper_forwards": 0,
            "teacher_14b_forwards": 0,
            "optimizer_steps": 0,
            "training_checkpoints_written": 0,
        }
    )
    write_json(lock_receipt_path, lock_receipt)

    summary_path = RUN_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "status": "locked_before_labeling",
            "publish_recovered_without_model_rerun": True,
            "private_drive_artifact_count_verified": len(verified_artifacts),
            "labeling_proper_started": False,
            "training_started": False,
        }
    )
    write_json(summary_path, summary)
    summary_md_path = RUN_DIR / "summary.md"
    summary_md_path.write_text(
        "# Paper Two D0 Preregistration Lock\n\n"
        "- Status: locked before labeling\n"
        "- Density inference: completed in the original pre-lock run\n"
        "- Publication recovery: metadata-only; no model rerun\n"
        f"- Private Drive artifacts verified: `{len(verified_artifacts)}`\n"
        f"- Code corpus: `{STACK_DATASET}` (Stack v1) at `{STACK_REVISION}`\n"
        "- Labeling proper: not started\n"
        "- Training: not started\n",
        encoding="utf-8",
    )
    for path in (
        manifest_path,
        preregistration_path,
        source_access_path,
        lock_receipt_path,
        summary_path,
        summary_md_path,
    ):
        copy_updated_receipt_to_drive(path)
    commit = publish()
    print(json.dumps({**summary, "lock_commit": commit}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
