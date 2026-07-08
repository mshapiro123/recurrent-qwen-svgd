"""Write receipts for the support-6 seed-replication frontier discrepancy."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json
from colab.stage5_frontier_metrics import (
    bar_crossing_frontier,
    deepest_passing_selection_frontier,
    diagonal_counts_to_accuracy,
    frontier_in_band,
)


DEFAULT_ORIGINAL_SUMMARY = "outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json"
DEFAULT_REPLICATION_SUMMARY = "outputs/stage5/stage5_support6_seed_replication_20260707_122930/summary.json"
FRONTIER_BAR = 0.71
TARGET_FRONTIER = 9.0
TOLERANCE = 1.0


def score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    score = payload.get("route_score") or {}
    selection = score.get("selection") or {}
    diagonal_counts = score.get("diagonal_counts") or {}
    diag_acc = diagonal_counts_to_accuracy(diagonal_counts)
    canonical = bar_crossing_frontier(diag_acc, bar=FRONTIER_BAR)
    return {
        "run_id": payload.get("run_id"),
        "summary": None,
        "status": payload.get("status"),
        "overall_pass": bool(score.get("overall_pass")),
        "canonical_frontier": canonical,
        "canonical_frontier_pass": frontier_in_band(canonical, target=TARGET_FRONTIER, tolerance=TOLERANCE),
        "deepest_passing_selection_frontier": deepest_passing_selection_frontier(selection),
        "active_diagonal": diag_acc,
        "diagonal_counts": diagonal_counts,
        "selected_correct": {
            str(depth): int(item.get("correct", 0))
            for depth, item in sorted(selection.items(), key=lambda item: int(item[0]))
            if isinstance(item, dict)
        },
    }


def selected_config(payload: dict[str, Any]) -> dict[str, Any]:
    train_config = payload.get("train_config") or {}
    init_metadata = payload.get("init_checkpoint_metadata") or {}
    data_summary = payload.get("data_summary") or {}
    data_config: dict[str, Any] = {}
    if isinstance(data_summary, dict):
        data_config = data_summary.get("config") or {}
    elif isinstance(data_summary, str):
        data_path = ROOT / data_summary
        if data_path.exists():
            raw_data_summary = json.loads(data_path.read_text(encoding="utf-8"))
            data_config = raw_data_summary.get("config") or {}
    return {
        "kind": payload.get("kind"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "init_source_summary": init_metadata.get("source_summary"),
        "init_source_run_id": init_metadata.get("source_run_id"),
        "init_source_stage_name": init_metadata.get("source_stage_name"),
        "init_source_checkpoint_drive_backup": init_metadata.get("source_checkpoint_drive_backup"),
        "train_max_depth": payload.get("train_max_depth"),
        "eval_max_depth": payload.get("eval_max_depth"),
        "max_depth": payload.get("max_depth"),
        "n_symbols": payload.get("n_symbols"),
        "threshold": payload.get("threshold"),
        "rows_per_depth": payload.get("rows_per_depth"),
        "heldout_rows_per_depth": payload.get("heldout_rows_per_depth"),
        "total_steps": payload.get("total_steps"),
        "ramp_steps": payload.get("ramp_steps"),
        "hold_steps": payload.get("hold_steps"),
        "loop_loss_mode": payload.get("loop_loss_mode"),
        "train_config": {
            key: train_config.get(key)
            for key in sorted(train_config)
            if key
            in {
                "batch_size",
                "learning_rate",
                "max_grad_norm",
                "max_length",
                "max_loops",
                "max_steps",
                "model_name",
                "optimizer",
                "split",
                "train_on_prompt",
                "weight_decay",
            }
        },
        "data_config": data_config if isinstance(data_config, dict) else {},
    }


def diff_against_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ignored = {"run_id", "status"}
    diff: dict[str, Any] = {}
    keys = sorted(set(reference) | set(candidate))
    for key in keys:
        if key in ignored:
            continue
        left = reference.get(key)
        right = candidate.get(key)
        if left != right:
            diff[key] = {"original": left, "candidate": right}
    return diff


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Support-6 Replication Receipts - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Canonical frontier: `bar_crossing_frontier`, bar `{FRONTIER_BAR}`",
        f"- Target band: `{TARGET_FRONTIER} +/- {TOLERANCE}`",
        "",
        "## Scores",
    ]
    for item in payload["runs"]:
        lines.extend(
            [
                "",
                f"### {item['label']}",
                f"- run_id: `{item['run_id']}`",
                f"- canonical_frontier: `{item['canonical_frontier']}`",
                f"- canonical_frontier_pass: `{item['canonical_frontier_pass']}`",
                f"- deepest_passing_selection_frontier: `{item['deepest_passing_selection_frontier']}`",
                f"- selected_correct: `{item['selected_correct']}`",
            ]
        )
    lines.extend(["", "## Config Diffs Against Original"])
    for diff in payload["config_diffs"]:
        lines.append(f"- `{diff['label']}`: `{diff['diff']}`")
    lines.extend(["", "## Decision"])
    lines.append(payload["decision"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SUPPORT6_RECEIPTS_RUN_ID") or time.strftime(
        "stage5_support6_replication_receipts_%Y%m%d_%H%M%S"
    )
    original_summary = os.environ.get("STAGE5_SUPPORT6_ORIGINAL_SUMMARY", DEFAULT_ORIGINAL_SUMMARY)
    replication_summary = os.environ.get("STAGE5_SUPPORT6_REPLICATION_SUMMARY", DEFAULT_REPLICATION_SUMMARY)
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    original = read_json(original_summary)
    replication = read_json(replication_summary)
    original_score = score_payload(original)
    original_score.update({"label": "original", "summary": original_summary})

    runs = [original_score]
    config_reference = selected_config(original)
    config_diffs: list[dict[str, Any]] = []
    for result in replication.get("results", []):
        child_summary = result.get("summary")
        child = read_json(child_summary)
        score = score_payload(child)
        score.update({"label": f"seed_{result.get('seed')}", "summary": child_summary})
        runs.append(score)
        config_diffs.append(
            {
                "label": f"seed_{result.get('seed')}",
                "summary": child_summary,
                "diff": diff_against_reference(config_reference, selected_config(child)),
            }
        )

    replicate_scores = [item for item in runs if item["label"] != "original"]
    replicate_passes = [bool(item["canonical_frontier_pass"]) for item in replicate_scores]
    if replicate_scores and all(replicate_passes):
        status = "replication_canonical_pass"
        decision = "All replicate seeds clear the preregistered canonical frontier band."
    else:
        status = "replication_needs_dosed_seed_resolution"
        decision = (
            "At least one replicate seed fails the canonical bar-crossing frontier band. "
            "Run the pre-registered dosed-seed resolution before using this cell as robustness evidence."
        )

    payload = {
        "kind": "stage5_support6_replication_receipts",
        "run_id": run_id,
        "status": status,
        "frontier_policy": {
            "canonical_function": "bar_crossing_frontier",
            "bar": FRONTIER_BAR,
            "target": TARGET_FRONTIER,
            "tolerance": TOLERANCE,
            "deprecated_metric": "deepest_passing_selection_frontier",
        },
        "original_summary": original_summary,
        "replication_summary": replication_summary,
        "runs": runs,
        "config_diffs": config_diffs,
        "decision": decision,
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    if os.environ.get("STAGE5_SUPPORT6_RECEIPTS_PUBLISH", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
        publish_run(run_dir, message=f"Record Stage 5 support-6 replication receipts {run_id} [skip ci]")
    else:
        print("Skipping publish because STAGE5_SUPPORT6_RECEIPTS_PUBLISH=0", flush=True)
    print(json.dumps({"run_id": run_id, "status": status, "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
