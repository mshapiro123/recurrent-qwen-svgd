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
RESEARCH_FOLDER_ID = "1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr"
COMPACT_FILE_NAME = "p34_share_compact_batch_20260812.pt"
COMPACT_DRIVE_ID = "1NvWNpSie9lYnNjpmoJrWldyFMMmKpiK6"
COMPACT_SHA256 = "d34be99c003e59364c80991cbdfb4ad698f499bc868d421c8f431cbe01799fb7"
SEED = int(os.environ.get("P34_SHARE_SEED", "0"))
if SEED not in (0, 1):
    raise ValueError("P34_SHARE_SEED must be 0 or 1")
OUTPUT = (
    ROOT
    / "outputs/stage5"
    / RUN_ID
    / f"receipts/p34_share_calibration_seed_{SEED}.json"
)

MIGRATED_SHA256 = {
    0: "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
    1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f",
}
P33_SHA256 = {
    0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
    1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
}
I1_SHA256 = {
    0: "01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88",
    1: "2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a",
}


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


def copy_research_file(file_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            RCLONE,
            "copyto",
            f"research:{file_name}",
            destination,
            "--config",
            RCLONE_CONFIG,
            "--drive-root-folder-id",
            RESEARCH_FOLDER_ID,
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
    old = json.loads(old_summary.read_text(encoding="utf-8"))
    compact = STAGE / "p34_share_compact_batch.pt"
    copy_research_file(COMPACT_FILE_NAME, compact)
    compact_sha256 = sha256_file(compact)
    if compact_sha256 != COMPACT_SHA256:
        raise RuntimeError(
            f"P3.4 compact batch hash mismatch for Drive file {COMPACT_DRIVE_ID}"
        )

    lm_head_receipt = old["model_caches"]["student_0p5b"]["lm_head"]
    lm_head = STAGE / "lm_head.pt"
    copy_file(
        "stage5_paper2_phase2_stage0a_20260803/private/stage0a/"
        + relative_private(str(lm_head_receipt["path"])),
        lm_head,
    )
    migrated = STAGE / f"seed_{SEED}_migrated.pt"
    p33 = STAGE / f"seed_{SEED}_p33.pt"
    i1 = STAGE / f"seed_{SEED}_i1.pt"
    copy_file(
        f"stage5_paper2_phase3_p31_p32_receipts_20260810/private/migrated_checkpoints/seed_{SEED}_full_a2_phase3_migrated.pt",
        migrated,
    )
    copy_file(
        f"stage5_paper2_phase3_p33_20260811/private/seed_{SEED}/checkpoint_step_1000.pt",
        p33,
    )
    copy_file(
        f"stage5_paper2_phase3_p33_i1_20260812/private/seed_{SEED}/resume.pt",
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
            "--compact_batch",
            compact,
            "--compact_batch_sha256",
            compact_sha256,
            "--lm_head",
            lm_head,
            "--lm_head_sha256",
            lm_head_receipt["sha256"],
            "--migrated",
            migrated,
            "--migrated_sha256",
            MIGRATED_SHA256[SEED],
            "--p33",
            p33,
            "--p33_sha256",
            P33_SHA256[SEED],
            "--i1",
            i1,
            "--i1_sha256",
            I1_SHA256[SEED],
            "--seed",
            str(SEED),
            *(["--main_only"] if SEED == 1 else []),
            "--output",
            OUTPUT,
            "--device",
            "cuda",
        ]
    )
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if result["optimizer_steps"] != 0 or result["training_authorized"]:
        raise RuntimeError("P3.4 share calibration crossed the no-training boundary")
    copy_file_target = f"{RUN_ID}/receipts/p34_share_calibration_seed_{SEED}.json"
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
