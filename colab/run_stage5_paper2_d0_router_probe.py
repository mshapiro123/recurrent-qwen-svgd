"""Run and publish the read-only D0 deployable-router probe."""

# Safety marker: read-only L4 feature extraction no model optimizer no model training

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

from colab.run_stage5_paper2_d0_floor_calibration import (
    DRIVE_ROOT,
    RUN_DIR,
    restore_calibration,
    restore_checkpoint,
)
from training.speculative_depth_d0_corpus import sha256_file


def run(command: list[str]) -> None:
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
    if code:
        raise subprocess.CalledProcessError(code, command)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 router probe produced no aggregate receipt changes")
    run(["git", "commit", "-m", "Record Paper Two D0 deployable router probe [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    floor_summary = read_json(RUN_DIR / "floor" / "summary.json")
    oracle_summary = read_json(RUN_DIR / "router_oracle_audit" / "summary.json")
    if floor_summary.get("status") != "complete" or floor_summary.get("training_started") is not False:
        raise RuntimeError("router probe requires the completed pretraining floor")
    if oracle_summary.get("status") != "complete":
        raise RuntimeError("router probe requires the completed oracle audit")
    oracle_uplift = float(oracle_summary["teacher_7b"]["oracle_any_depth"]["uplift_over_best_fixed"])
    if oracle_uplift < 0.01:
        raise RuntimeError(
            f"router probe is not authorized: oracle uplift {oracle_uplift:.6f} is below 0.01"
        )
    calibration = restore_calibration()
    checkpoint, resolution = restore_checkpoint()
    private_rows = DRIVE_ROOT / "private" / "floor" / "floor_rows.json"
    expected_private = str(floor_summary["private_rows_sha256"])
    if not private_rows.exists() or sha256_file(private_rows) != expected_private:
        raise RuntimeError("router probe private floor rows are missing or hash-mismatched")
    print(
        "router_probe_preflight:",
        json.dumps(
            {
                "oracle_uplift": oracle_uplift,
                "checkpoint": str(checkpoint),
                "checkpoint_resolution": resolution,
                "private_rows_sha256": expected_private,
                "evaluation_partition_touched": False,
                "teacher_features_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    private_root = DRIVE_ROOT / "private" / "router_probe"
    output_dir = RUN_DIR / "router_probe"
    summary_path = output_dir / "summary.json"
    run(
        [
            sys.executable,
            "eval/eval_speculative_depth_router_probe.py",
            "--data_jsonl",
            str(calibration),
            "--floor_private_rows",
            str(private_rows),
            "--checkpoint",
            str(checkpoint),
            "--feature_cache",
            str(private_root / "feature_cache.pt"),
            "--row_cache_dir",
            str(private_root / "row_cache"),
            "--output_summary",
            str(summary_path),
            "--expected_private_rows_sha256",
            expected_private,
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_D0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
            "--projection_dim",
            os.environ.get("STAGE5_D0_ROUTER_PROJECTION_DIM", "128"),
            "--seed",
            os.environ.get("STAGE5_D0_ROUTER_SEED", "20260727"),
        ]
    )
    summary = read_json(summary_path)
    if not summary.get("no_model_training") or not summary.get("no_model_mutation"):
        raise RuntimeError("router probe violated its frozen-model contract")
    drive_receipt = DRIVE_ROOT / "receipts" / "router_probe_summary.json"
    drive_receipt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, drive_receipt)
    markdown = output_dir / "summary.md"
    markdown.write_text(
        "# Paper Two D0 Deployable Router Probe\n\n"
        f"- Prelude any-extra-depth AUROC: `{summary['preloop']['any_extra_depth']['hidden_plus_structure']['test_auroc']:.6f}`\n"
        f"- Prelude loop-2 decision AUROC: `{summary['preloop']['loop2_decision']['hidden_plus_structure']['test_auroc']:.6f}`\n"
        f"- Loop-2 structural baseline AUROC: `{summary['preloop']['loop2_decision']['structure_only']['test_auroc']:.6f}`\n"
        f"- Pre-loop verdict: `{summary['preloop']['verdict']}`\n"
        "- Teacher features used: `false`\n"
        "- Evaluation partition touched: `false`\n"
        "- Model training or mutation: `none`\n",
        encoding="utf-8",
    )
    commit = publish([summary_path, markdown])
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
