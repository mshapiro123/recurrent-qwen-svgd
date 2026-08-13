"""Stage and run one resumable, lock-bound P3.4 seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p34_20260813"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
PREFLIGHT_ID = "stage5_paper2_phase3_retention_preflight_20260811"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
MIGRATION_ID = "stage5_paper2_phase3_p31_p32_receipts_20260810"
ORACLE_ID = "stage5_paper2_phase3_oracle_forecast_20260810"
P33_ID = "stage5_paper2_phase3_p33_20260811"
I1_ID = "stage5_paper2_phase3_p33_i1_20260812"
MIGRATED_SHA = {
    0: "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
    1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f",
}
P33_SHA = {
    0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
    1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
}
PREFLIGHT_TRANSPORT_PARTS = (
    (
        "p33_retention_preflight.zip.part01",
        "c783a473b0647d6d4902e08de6ca560488d9a3058c36f5b8b8ee978fcd3c4068",
    ),
    (
        "p33_retention_preflight.zip.part02",
        "7382e1076542978a8160dc98ea7e2c44150f3c88999aa91299cfac53d6242991",
    ),
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> None:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)


def rsync(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run([
        "rsync", "--archive", "--delete", "--partial", "--info=progress2",
        str(source) + (os.sep if source.is_dir() else ""),
        str(destination) + (os.sep if source.is_dir() else ""),
    ])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_preflight(scratch: Path) -> Path:
    canonical = DRIVE_STAGE5 / PREFLIGHT_ID
    staged_labels = canonical / "private/p33_prep/p33_staged_labels.jsonl"
    if staged_labels.exists():
        return canonical

    transport = scratch / "preflight_transport"
    extracted = transport / "extracted"
    extracted_labels = extracted / "drive/private/p33_prep/p33_staged_labels.jsonl"
    if extracted_labels.exists():
        return extracted / "drive"

    transport.mkdir(parents=True, exist_ok=True)
    archive = transport / "preflight.zip"
    with archive.open("wb") as output:
        for name, expected_sha256 in PREFLIGHT_TRANSPORT_PARTS:
            part = Path("/content/drive/MyDrive") / name
            observed_sha256 = sha256_file(part)
            if observed_sha256 != expected_sha256:
                raise RuntimeError(
                    f"P3.3 preflight transport SHA mismatch for {name}: "
                    f"expected={expected_sha256} observed={observed_sha256}"
                )
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    with zipfile.ZipFile(archive) as bundle:
        bundle.testzip()
        bundle.extractall(extracted)
    archive.unlink()
    if not extracted_labels.exists():
        raise RuntimeError("P3.3 preflight transport omitted staged labels")
    return extracted / "drive"


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 80 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("P3.3 requires at least 80 GiB local scratch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=("main", "slot"), required=True)
    args = parser.parse_args()
    seed = args.seed
    arm = args.arm
    if arm == "slot" and seed != 0:
        raise ValueError("P3.4 slot arm is authorized for seed 0 only")
    label = f"{arm}_seed_{seed}"
    run_dir = ROOT / "outputs/stage5" / RUN_ID / label
    drive_run = DRIVE_STAGE5 / RUN_ID
    private = drive_run / "private" / label
    receipts = drive_run / "receipts" / label
    status_path = receipts / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(status_path, {
            "kind": "paper2_phase3_p34_colab_status_v1", "seed": seed, "arm": arm,
            "status": value, "updated_at_unix": time.time(), **details,
        })
        print(f"p34_status arm={arm} seed={seed} status={value} details={details}", flush=True)

    try:
        status("staging")
        scratch = scratch_root()
        old = scratch / "old"
        new = scratch / "new"
        preflight = resolve_preflight(scratch)
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/sample_manifest.jsonl", old / "sample_manifest.jsonl")
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/lattice", old / "lattice")
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/model_cache/student_0p5b", old / "model_cache/student_0p5b")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/sample_manifest.jsonl", new / "sample_manifest.jsonl")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/lattice", new / "lattice")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/model_cache/student_0p5b", new / "model_cache/student_0p5b")
        direction_cache = scratch / "agreement_oracle_directions.pt"
        migrated = scratch / f"seed_{seed}_migrated.pt"
        p33_checkpoint = scratch / f"seed_{seed}_p33_step_1000.pt"
        i1_checkpoint = scratch / f"seed_{seed}_i1.pt"
        rsync(
            DRIVE_STAGE5 / ORACLE_ID / "private/oracle_cache/agreement_oracle_directions.pt",
            direction_cache,
        )
        rsync(
            DRIVE_STAGE5 / MIGRATION_ID / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt",
            migrated,
        )
        rsync(
            DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt",
            p33_checkpoint,
        )
        rsync(DRIVE_STAGE5 / I1_ID / f"private/seed_{seed}/resume.pt", i1_checkpoint)
        status("training")
        command = [
            sys.executable, "-u", "-m", "training.run_paper2_phase3_p34",
            "--seed", str(seed),
            "--arm", arm,
            "--old_summary", str(ROOT / "outputs/stage5" / OLD_ID / "summary.json"),
            "--old_private", str(old),
            "--new_summary", str(DRIVE_STAGE5 / NEW_ID / "receipts/full_cache_summary.json"),
            "--new_private", str(new),
            "--staged_labels", str(preflight / "private/p33_prep/p33_staged_labels.jsonl"),
            "--positive_audit", str(preflight / "private/p33_prep/p33_audit_slice.jsonl"),
            "--negative_audit", str(preflight / "private/p33_prep/p33_negative_audit_slice.jsonl"),
            "--retention_panel", str(preflight / "private/p33_prep/p33_retention_panel.jsonl"),
            "--direction_cache", str(direction_cache),
            "--dev_panel", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"),
            "--base_scores", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"),
            "--share_rows", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/share_calibration/p34_share_calibration_rows.jsonl"),
            "--migrated", str(migrated), "--migrated_sha256", MIGRATED_SHA[seed],
            "--p33", str(p33_checkpoint), "--p33_sha256", P33_SHA[seed],
            "--i1", str(i1_checkpoint),
            "--lock", str(ROOT / "training/paper2_phase3_p34_preregistration.json"),
            "--output_dir", str(run_dir),
            "--private_dir", str(private),
            "--device", "cuda",
        ]
        run(command, allowed=(0, 2))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        write_json(receipts / "summary.json", summary)
        status("complete", step=summary["step"], run_status=summary["status"])
        return 0
    except Exception as error:
        status("failed", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())

