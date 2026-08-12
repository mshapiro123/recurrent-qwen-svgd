"""Stage selected Drive shards and run the read-only P3.4 share calibration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path("/content/recurrent-qwen-svgd")
RUN_ID = "stage5_paper2_phase3_p34_lock_20260812"
DRIVE_REMOTE = os.environ.get(
    "P34_DRIVE_REMOTE", "research:recurrent-qwen-svgd-artifacts/stage5"
)
RCLONE = Path(os.environ.get("P34_RCLONE", "/content/bin/rclone"))
RCLONE_CONFIG = Path(os.environ.get("P34_RCLONE_CONFIG", "/content/rclone.conf"))
STAGE = Path("/mnt/local-scratch/p34-share-calibration")
OUTPUT = ROOT / "outputs/stage5" / RUN_ID / "receipts/p34_share_calibration.json"


def run(command: Sequence[str], *, cwd: Path = ROOT) -> None:
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(remote_relative: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    run(
        [
            RCLONE,
            "copyto",
            f"{DRIVE_REMOTE}/{remote_relative}",
            destination,
            "--config",
            RCLONE_CONFIG,
            "--transfers",
            "8",
        ],
        cwd=Path("/content"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def student_receipts(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(summary["model_caches"]["student_0p5b"]["shards"])


def relative_private(path: str) -> str:
    normalized = path.replace("\\", "/")
    for marker in ("/private/stage0a/", "/private/full/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    raise RuntimeError(f"P3.4 cannot relativize private cache path: {path}")


def required_receipt_indexes(
    *, receipts: Sequence[Mapping[str, Any]], needed_sample_indices: set[int]
) -> list[int]:
    selected = []
    start = 0
    for index, receipt in enumerate(receipts):
        stop = start + int(receipt["samples"])
        if any(start <= value < stop for value in needed_sample_indices):
            selected.append(index)
        start = stop
    if not selected or max(needed_sample_indices) >= start:
        raise RuntimeError("P3.4 selected sample index is outside the shard ledger")
    return selected


def stage_source(
    *,
    source: str,
    run_id: str,
    private_prefix: str,
    summary: Mapping[str, Any],
    summary_path: Path,
    manifest_path: Path,
    records: Sequence[Mapping[str, Any]],
    private_root: Path,
) -> dict[str, Any]:
    private_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, private_root / "sample_manifest.jsonl")
    selected_anchors = {
        int(row["anchor_index"]) for row in records if str(row["source"]) == source
    }
    manifest = read_jsonl(manifest_path)
    needed_indices = {
        int(row["sample_index"])
        for row in manifest
        if int(row["anchor_index"]) in selected_anchors
    }
    if len(needed_indices) != 4 * len(selected_anchors):
        raise RuntimeError(f"P3.4 {source} selected anchors do not have four horizons")
    lattice = list(summary["lattice"]["shards"])
    student = student_receipts(summary)
    if len(lattice) != len(student):
        raise RuntimeError("P3.4 source shard ledgers differ")
    indexes = required_receipt_indexes(
        receipts=lattice, needed_sample_indices=needed_indices
    )
    copied = []
    for index in indexes:
        for receipt in (lattice[index], student[index]):
            relative = relative_private(str(receipt["path"]))
            copy_file(
                f"{run_id}/{private_prefix}/{relative}", private_root / relative
            )
            copied.append({"relative": relative, "sha256": receipt["sha256"]})
    return {
        "source": source,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_anchors": len(selected_anchors),
        "selected_sample_indices": len(needed_indices),
        "selected_shards": len(indexes),
        "copied_file_ledger_sha256": hashlib.sha256(
            json.dumps(copied, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    if not RCLONE_CONFIG.is_file():
        raise RuntimeError("P3.4 share calibration needs the read-only rclone config")
    STAGE.mkdir(parents=True, exist_ok=True)
    rows = ROOT / f"outputs/stage5/{RUN_ID}/share_calibration/p34_share_calibration_rows.jsonl"
    selection = ROOT / f"outputs/stage5/{RUN_ID}/receipts/p34_share_calibration_selection.json"
    records = read_jsonl(rows)

    old_summary = ROOT / "outputs/stage5/stage5_paper2_phase2_stage0a_20260803/summary.json"
    new_summary = STAGE / "new_summary.json"
    old_manifest = STAGE / "old_manifest.jsonl"
    new_manifest = STAGE / "new_manifest.jsonl"
    copy_file(
        "stage5_paper2_phase2_option_b_teacher_cache_20260806/receipts/full_cache_summary.json",
        new_summary,
    )
    copy_file(
        "stage5_paper2_phase2_stage0a_20260803/private/stage0a/sample_manifest.jsonl",
        old_manifest,
    )
    copy_file(
        "stage5_paper2_phase2_option_b_teacher_cache_20260806/private/full/sample_manifest.jsonl",
        new_manifest,
    )
    old = json.loads(old_summary.read_text(encoding="utf-8"))
    new = json.loads(new_summary.read_text(encoding="utf-8"))
    old_private = STAGE / "old"
    new_private = STAGE / "new"
    staging = [
        stage_source(
            source="old",
            run_id="stage5_paper2_phase2_stage0a_20260803",
            private_prefix="private/stage0a",
            summary=old,
            summary_path=old_summary,
            manifest_path=old_manifest,
            records=records,
            private_root=old_private,
        ),
        stage_source(
            source="new",
            run_id="stage5_paper2_phase2_option_b_teacher_cache_20260806",
            private_prefix="private/full",
            summary=new,
            summary_path=new_summary,
            manifest_path=new_manifest,
            records=records,
            private_root=new_private,
        ),
    ]
    staging_path = STAGE / "staging_receipt.json"
    staging_path.write_text(json.dumps(staging, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lm_head_receipt = old["model_caches"]["student_0p5b"]["lm_head"]
    lm_head = STAGE / "lm_head.pt"
    copy_file(
        "stage5_paper2_phase2_stage0a_20260803/private/stage0a/"
        + relative_private(str(lm_head_receipt["path"])),
        lm_head,
    )
    direction = STAGE / "agreement_oracle_directions.pt"
    migrated = STAGE / "seed_0_migrated.pt"
    p33 = STAGE / "seed_0_p33.pt"
    i1 = STAGE / "seed_0_i1.pt"
    copy_file(
        "stage5_paper2_phase3_oracle_forecast_20260810/private/oracle_cache/agreement_oracle_directions.pt",
        direction,
    )
    copy_file(
        "stage5_paper2_phase3_p31_p32_receipts_20260810/private/migrated_checkpoints/seed_0_full_a2_phase3_migrated.pt",
        migrated,
    )
    copy_file(
        "stage5_paper2_phase3_p33_20260811/private/seed_0/checkpoint_step_1000.pt",
        p33,
    )
    copy_file(
        "stage5_paper2_phase3_p33_i1_20260812/private/seed_0/resume.pt",
        i1,
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p34_share_calibration",
            "--rows",
            rows,
            "--selection_receipt",
            selection,
            "--old_summary",
            old_summary,
            "--old_private",
            old_private,
            "--new_summary",
            new_summary,
            "--new_private",
            new_private,
            "--lm_head",
            lm_head,
            "--lm_head_sha256",
            lm_head_receipt["sha256"],
            "--direction_cache",
            direction,
            "--migrated",
            migrated,
            "--migrated_sha256",
            "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
            "--p33",
            p33,
            "--p33_sha256",
            "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
            "--i1",
            i1,
            "--i1_sha256",
            "01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88",
            "--output",
            OUTPUT,
            "--device",
            "cuda",
        ]
    )
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if result["optimizer_steps"] != 0 or result["training_authorized"]:
        raise RuntimeError("P3.4 share calibration crossed the no-training boundary")
    copy_file_target = f"{RUN_ID}/receipts/p34_share_calibration.json"
    run(
        [
            RCLONE,
            "copyto",
            OUTPUT,
            f"{DRIVE_REMOTE}/{copy_file_target}",
            "--config",
            RCLONE_CONFIG,
        ],
        cwd=Path("/content"),
    )
    print(f"P34_SHARE_CALIBRATION_RECEIPT={OUTPUT}", flush=True)
    print(f"P34_SHARE_CALIBRATION_SHA256={sha256_file(OUTPUT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
