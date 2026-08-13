"""Create the immutable CPU compact batch for P3.4 share calibration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from colab.run_stage5_paper2_phase3_p34_share_calibration import (
    DRIVE_REMOTE,
    RCLONE,
    RCLONE_CONFIG,
    ROOT,
    RUN_ID,
    STAGE,
    copy_file,
    read_jsonl,
    run,
    sha256_file,
    stage_source,
)
from eval.eval_paper2_phase3_p34_share_calibration import (
    _selected_source_cache,
    build_selected_batch,
)


OUTPUT = STAGE / "p34_share_compact_batch.pt"
RECEIPT = ROOT / "outputs/stage5" / RUN_ID / "receipts/p34_share_compaction.json"


def main() -> int:
    if not RCLONE_CONFIG.is_file():
        raise RuntimeError("P3.4 share compaction needs the read-only rclone config")
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
    stage_source(
        source="old",
        run_id="stage5_paper2_phase2_stage0a_20260803",
        private_prefix="private/stage0a",
        summary=old,
        summary_path=old_summary,
        manifest_path=old_manifest,
        records=records,
        private_root=old_private,
    )
    stage_source(
        source="new",
        run_id="stage5_paper2_phase2_option_b_teacher_cache_20260806",
        private_prefix="private/full",
        summary=new,
        summary_path=new_summary,
        manifest_path=new_manifest,
        records=records,
        private_root=new_private,
    )
    direction = STAGE / "agreement_oracle_directions.pt"
    copy_file(
        "stage5_paper2_phase3_oracle_forecast_20260810/private/oracle_cache/agreement_oracle_directions.pt",
        direction,
    )
    sources = {}
    source_receipts = []
    for name, summary_path, private_root in (
        ("old", old_summary, old_private),
        ("new", new_summary, new_private),
    ):
        sources[name], source_receipt = _selected_source_cache(
            source=name,
            records=records,
            summary_path=summary_path,
            private_root=private_root,
        )
        source_receipts.append(source_receipt)
    batch, batch_receipt = build_selected_batch(
        records=records,
        sources=sources,
        direction_cache=direction,
        device="cpu",
    )
    payload = {
        "kind": "paper2_phase3_p34_compact_share_batch_v1",
        "selection_receipt_sha256": sha256_file(selection),
        "rows_file_sha256": sha256_file(rows),
        "source_receipts": source_receipts,
        "batch_receipt": batch_receipt,
        "batch": {key: value.cpu() for key, value in batch.items()},
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    temporary = OUTPUT.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(OUTPUT)
    receipt = {
        key: value for key, value in payload.items() if key != "batch"
    }
    receipt.update(
        {
            "compact_batch_sha256": sha256_file(OUTPUT),
            "compact_batch_bytes": OUTPUT.stat().st_size,
        }
    )
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for local, remote_name in (
        (OUTPUT, "private/p34_share_compact_batch.pt"),
        (RECEIPT, "receipts/p34_share_compaction.json"),
    ):
        run(
            [
                RCLONE,
                "copyto",
                local,
                f"{DRIVE_REMOTE}/{RUN_ID}/{remote_name}",
                "--config",
                RCLONE_CONFIG,
            ],
            cwd=Path("/content"),
        )
    print(f"P34_SHARE_COMPACT_BATCH={OUTPUT}", flush=True)
    print(f"P34_SHARE_COMPACT_SHA256={receipt['compact_batch_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
