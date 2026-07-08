"""Continue failed support-6 replicate seeds for a fixed-dose resolution run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_support6_replication_receipts import (  # noqa: E402
    FRONTIER_BAR,
    TARGET_FRONTIER,
    TOLERANCE,
    score_payload,
)
from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json  # noqa: E402
from colab.stage5_frontier_metrics import frontier_in_band  # noqa: E402


DEFAULT_RECEIPT_SUMMARY = "outputs/stage5/stage5_support6_replication_receipts_20260708_003055/summary.json"


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def seed_from_label(label: str) -> str:
    prefix = "seed_"
    if label.startswith(prefix):
        return label[len(prefix) :]
    raise ValueError(f"Cannot infer seed from receipt label: {label!r}")


def failed_replicate_runs(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for item in receipt.get("runs") or []:
        label = str(item.get("label") or "")
        if label == "original":
            continue
        if not bool(item.get("canonical_frontier_pass")):
            failed.append(item)
    return failed


def summarize_dosed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"status": "no_failed_replicates_to_resolve"}
    frontier_passes = [
        frontier_in_band(
            float(item["post_dose"]["canonical_frontier"]),
            target=TARGET_FRONTIER,
            tolerance=TOLERANCE,
        )
        for item in results
        if item.get("post_dose", {}).get("canonical_frontier") is not None
    ]
    improved = [
        float(item["post_dose"]["canonical_frontier"]) > float(item["pre_dose"]["canonical_frontier"])
        for item in results
        if item.get("post_dose", {}).get("canonical_frontier") is not None
        and item.get("pre_dose", {}).get("canonical_frontier") is not None
    ]
    return {
        "frontier_policy": "bar_crossing_frontier",
        "frontier_bar": FRONTIER_BAR,
        "target_frontier": TARGET_FRONTIER,
        "frontier_tolerance": TOLERANCE,
        "all_dosed_frontiers_in_band": bool(frontier_passes) and all(frontier_passes),
        "all_dosed_frontiers_improved": bool(improved) and all(improved),
        "post_dose_frontiers": [item["post_dose"]["canonical_frontier"] for item in results],
        "status": "dosed_seed_resolution_pass"
        if frontier_passes and all(frontier_passes)
        else "dosed_seed_resolution_needs_review",
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Support-6 Dosed Seed Resolution - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source receipt: `{payload['source_receipt_summary']}`",
        f"- Extra steps per failed replicate: `{payload['extra_steps']}`",
        f"- Frontier policy: `bar_crossing_frontier`, bar `{FRONTIER_BAR}`",
        f"- Target band: `{TARGET_FRONTIER} +/- {TOLERANCE}`",
        "",
        "## Results",
    ]
    for result in payload.get("results", []):
        lines.extend(
            [
                "",
                f"### {result['label']}",
                f"- Seed: `{result['seed']}`",
                f"- Pre-dose frontier: `{result['pre_dose'].get('canonical_frontier')}`",
                f"- Post-dose frontier: `{result['post_dose'].get('canonical_frontier')}`",
                f"- Post-dose pass: `{result['post_dose'].get('canonical_frontier_pass')}`",
                f"- Child summary: `{result['child_summary']}`",
            ]
        )
    lines.extend(["", "## Decision", payload.get("decision", "")])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SUPPORT6_DOSED_RUN_ID") or time.strftime(
        "stage5_support6_dosed_seed_resolution_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_summary = os.environ.get("STAGE5_SUPPORT6_DOSED_RECEIPT_SUMMARY", DEFAULT_RECEIPT_SUMMARY)
    receipt = read_json(receipt_summary)
    extra_steps = int(os.environ.get("STAGE5_SUPPORT6_DOSED_STEPS", "2000"))
    failed = failed_replicate_runs(receipt)
    payload: dict[str, Any] = {
        "kind": "stage5_support6_dosed_seed_resolution",
        "run_id": run_id,
        "status": "started",
        "source_receipt_summary": receipt_summary,
        "extra_steps": extra_steps,
        "failed_replicates": failed,
        "results": [],
        "decision": "Dosed seed resolution is running.",
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    if not failed:
        payload.update(
            {
                "status": "no_failed_replicates_to_resolve",
                "resolution_summary": summarize_dosed_results([]),
                "decision": "The source receipt has no failed replicate seeds under the canonical frontier policy.",
            }
        )
        write_json(run_dir / "summary.json", payload)
        write_markdown(run_dir, payload)
        publish_run(run_dir, message=f"Record Stage 5 support-6 dosed seed resolution {run_id} [skip ci]")
        return 0

    for item in failed:
        label = str(item["label"])
        seed = seed_from_label(label)
        source_summary = str(item["summary"])
        child_id = f"{run_id}_{label}_dose{extra_steps}"
        run(
            [sys.executable, "colab/run_stage5_depth_support_route_comparison.py"],
            env={
                "STAGE5_ROUTE_RUN_ID": child_id,
                "STAGE5_ROUTE_INIT_CHECKPOINT": source_summary,
                "STAGE5_ROUTE_TRAIN_SEED": seed,
                "STAGE5_ROUTE_TOTAL_STEPS": str(extra_steps),
                "STAGE5_ROUTE_TRAIN_MAX_DEPTH": "6",
                "STAGE5_ROUTE_EVAL_MAX_DEPTH": "10",
                "STAGE5_ROUTE_ROWS_PER_DEPTH": os.environ.get("STAGE5_SUPPORT6_DOSED_ROWS_PER_DEPTH", "256"),
                "STAGE5_ROUTE_FROZEN_ROWS_PER_DEPTH": "128",
                "STAGE5_ROUTE_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v1",
                "STAGE5_ROUTE_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                    "STAGE5_SUPPORT6_DOSED_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
                ),
                "STAGE5_ROUTE_DTYPE": os.environ.get("STAGE5_SUPPORT6_DOSED_DTYPE", "bfloat16"),
            },
        )
        child_summary = ROOT / "outputs" / "stage5" / child_id / "summary.json"
        child = read_json(child_summary)
        post_dose = score_payload(child)
        result = {
            "label": label,
            "seed": seed,
            "pre_dose": item,
            "source_summary": source_summary,
            "child_summary": path_for_cli(child_summary),
            "post_dose": post_dose,
        }
        payload["results"].append(result)
        resolution = summarize_dosed_results(payload["results"])
        payload.update(
            {
                "status": resolution["status"],
                "resolution_summary": resolution,
                "decision": (
                    "All completed dosed failed-seed frontiers are back inside the preregistered target band."
                    if resolution["status"] == "dosed_seed_resolution_pass"
                    else "At least one completed dosed failed-seed frontier remains outside the target band."
                ),
            }
        )
        write_json(run_dir / "summary.json", payload)
        write_markdown(run_dir, payload)
        publish_run(run_dir, message=f"Record Stage 5 support-6 dosed seed partial {run_id} {label} [skip ci]")

    resolution = summarize_dosed_results(payload["results"])
    payload.update(
        {
            "status": resolution["status"],
            "resolution_summary": resolution,
            "decision": (
                "Dosed seed resolution passed: every failed replicate seed returned to the canonical frontier band."
                if resolution["status"] == "dosed_seed_resolution_pass"
                else "Dosed seed resolution did not fully pass; support-6 robustness remains seed-sensitive."
            ),
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 support-6 dosed seed resolution {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
