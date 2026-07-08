"""Run two additional support-6 route seeds for paper-grade replication."""

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

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json
from colab.stage5_frontier_metrics import (
    bar_crossing_frontier,
    deepest_passing_selection_frontier,
    diagonal_counts_to_accuracy,
    frontier_in_band,
)


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


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def canonical_frontier_from_score(score: dict[str, Any], *, bar: float = 0.71) -> float:
    return bar_crossing_frontier(diagonal_counts_to_accuracy(score.get("diagonal_counts") or {}), bar=bar)


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Support-6 Seed Replication - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- Frontier band target: `within +/-1 depth of 9`",
        "",
        "## Results",
    ]
    for result in payload.get("results", []):
        lines.append(
            f"- seed `{result['seed']}`: canonical_frontier=`{result.get('canonical_frontier')}`, "
            f"deepest_passing_selection_frontier=`{result.get('deepest_passing_selection_frontier')}`, "
            f"overall_pass=`{result.get('overall_pass')}`, summary=`{result.get('summary')}`"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    frontiers = [
        float(item["canonical_frontier"])
        for item in results
        if item.get("canonical_frontier") is not None
    ]
    if not frontiers:
        return {"status": "incomplete_no_frontiers"}
    target = float(os.environ.get("STAGE5_SUPPORT6_REPLICATION_TARGET_FRONTIER", "9"))
    within_band = [frontier_in_band(frontier, target=target, tolerance=1.0) for frontier in frontiers]
    return {
        "target_frontier": target,
        "frontier_definition": "bar_crossing_frontier",
        "frontier_bar": 0.71,
        "frontier_tolerance": 1.0,
        "frontiers": frontiers,
        "min_frontier": min(frontiers),
        "max_frontier": max(frontiers),
        "mean_frontier": sum(frontiers) / len(frontiers),
        "within_plus_minus_one": all(within_band),
        "status": "replication_pass" if all(within_band) else "replication_needs_review",
    }


def main() -> int:
    run_id = os.environ.get("STAGE5_SUPPORT6_REPLICATION_RUN_ID") or time.strftime(
        "stage5_support6_seed_replication_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(os.environ.get("STAGE5_SUPPORT6_REPLICATION_SEEDS", "20260716,20260726"))
    payload: dict[str, Any] = {
        "kind": "stage5_support6_seed_replication",
        "run_id": run_id,
        "status": "started",
        "seeds": seeds,
        "pre_registered_bar": "bar-crossing frontier within +/-1 depth of 9 across added seeds",
        "frontier_definition": "bar_crossing_frontier",
        "deprecated_frontier_definition": "deepest_passing_selection_frontier",
        "results": [],
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    for seed in seeds:
        child_id = f"{run_id}_seed{seed}"
        run(
            [sys.executable, "colab/run_stage5_depth_support_route_comparison.py"],
            env={
                "STAGE5_ROUTE_RUN_ID": child_id,
                "STAGE5_ROUTE_TRAIN_SEED": str(seed),
                "STAGE5_ROUTE_TOTAL_STEPS": os.environ.get("STAGE5_SUPPORT6_REPLICATION_STEPS", "2000"),
                "STAGE5_ROUTE_TRAIN_MAX_DEPTH": "6",
                "STAGE5_ROUTE_EVAL_MAX_DEPTH": "10",
                "STAGE5_ROUTE_ROWS_PER_DEPTH": os.environ.get("STAGE5_SUPPORT6_REPLICATION_ROWS_PER_DEPTH", "256"),
                "STAGE5_ROUTE_FROZEN_ROWS_PER_DEPTH": "128",
                "STAGE5_ROUTE_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v1",
            },
        )
        child_summary = ROOT / "outputs" / "stage5" / child_id / "summary.json"
        child = read_json(child_summary)
        score = child.get("route_score") or {}
        selection = score.get("selection") or {}
        selected = {depth: item.get("correct", 0) for depth, item in selection.items() if isinstance(item, dict)}
        canonical_frontier = canonical_frontier_from_score(score)
        result = {
            "seed": seed,
            "summary": path_for_cli(child_summary),
            "overall_pass": bool(score.get("overall_pass")),
            "selected_correct": selected,
            "canonical_frontier": canonical_frontier,
            "deepest_passing_selection_frontier": deepest_passing_selection_frontier(selection),
        }
        payload["results"].append(result)
        payload["replication_summary"] = summarize_results(payload["results"])
        payload["status"] = payload["replication_summary"]["status"]
        write_json(run_dir / "summary.json", payload)
        write_markdown(run_dir, payload)
        publish_run(run_dir, message=f"Record Stage 5 support-6 replication partial {run_id} seed {seed} [skip ci]")

    payload["replication_summary"] = summarize_results(payload["results"])
    payload["status"] = payload["replication_summary"]["status"]
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 support-6 seed replication {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
