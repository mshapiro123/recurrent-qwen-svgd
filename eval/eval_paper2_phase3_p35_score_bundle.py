"""Complete the registered raw/EMA and fixed-ceiling P3.5 score bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from training.paper2_phase3_p35 import (
    P35_PRIMARY_EVAL_CEILING,
    P35_SECONDARY_EVAL_CEILING,
    margin_summary,
)
from training.paper2_phase3_p33_prep import sha256_file
from training.run_paper2_phase3_p33 import read_jsonl, write_json


def paired_read(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fixes = sum(not bool(row["base_correct"]) and bool(row["augmented_correct"]) for row in rows)
    regressions = sum(bool(row["base_correct"]) and not bool(row["augmented_correct"]) for row in rows)
    by_battery: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_battery.setdefault(str(row["battery"]), []).append(row)
    return {
        "rows": len(rows),
        "correct": sum(bool(row["augmented_correct"]) for row in rows),
        "fixes": fixes,
        "regressions": regressions,
        "net_rows": fixes - regressions,
        "margins": margin_summary(rows),
        "by_battery": {
            name: {
                "rows": len(selected),
                "net_rows": sum(
                    int(bool(row["augmented_correct"])) - int(bool(row["base_correct"]))
                    for row in selected
                ),
            }
            for name, selected in sorted(by_battery.items())
        },
    }


def churn(left: list[Mapping[str, Any]], right: list[Mapping[str, Any]]) -> dict[str, Any]:
    left_by_id = {str(row["item_id"]): row for row in left}
    right_by_id = {str(row["item_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id) or len(left_by_id) != 1024:
        raise RuntimeError("P3.5 churn comparison row identity changed")
    changed = [
        item_id
        for item_id in sorted(left_by_id)
        if bool(left_by_id[item_id]["augmented_correct"])
        != bool(right_by_id[item_id]["augmented_correct"])
    ]
    left_fixes = {
        item_id for item_id, row in left_by_id.items()
        if not bool(row["base_correct"]) and bool(row["augmented_correct"])
    }
    right_fixes = {
        item_id for item_id, row in right_by_id.items()
        if not bool(row["base_correct"]) and bool(row["augmented_correct"])
    }
    union = left_fixes | right_fixes
    return {
        "changed_rows": len(changed),
        "changed_row_ids_sha256": __import__("hashlib").sha256(
            "\n".join(changed).encode()
        ).hexdigest(),
        "fix_set_jaccard": len(left_fixes & right_fixes) / max(1, len(union)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_summary", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    for name in ("migrated", "p33", "i1", "p34"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}_sha256", required=True)
    args = parser.parse_args()

    run = json.loads(args.run_summary.read_text(encoding="utf-8"))
    if run.get("kind") != "paper2_phase3_p35_landing_v1" or run.get("status") != "complete":
        raise RuntimeError("P3.5 score bundle requires a complete landing run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_reader = "probe" if run["arm"] == "probe_reader" else "mean"
    conditions: dict[str, dict[str, Any]] = {}
    row_sets: dict[str, list[dict[str, Any]]] = {}
    for entry in run["history"]:
        for variant in ("raw", "ema"):
            checkpoint = Path(entry[f"{variant}_checkpoint"]["path"])
            checkpoint_sha = str(entry[f"{variant}_checkpoint"]["sha256"])
            for ceiling in (P35_PRIMARY_EVAL_CEILING, P35_SECONDARY_EVAL_CEILING):
                label = f"step_{entry['step']}_{variant}_ceiling_{ceiling:.2f}"
                rows_path = args.output_dir / f"{label}.jsonl"
                summary_path = args.output_dir / f"{label}.json"
                command = [
                    sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p34_task_trajectory",
                    "--panel", str(args.panel), "--base_scores", str(args.base_scores),
                    "--output_jsonl", str(rows_path), "--summary", str(summary_path),
                    "--condition", f"p35_{run['arm']}_{variant}_seed_{run['seed']}",
                    "--look", str(entry["look"]), "--seed", str(run["seed"]),
                    "--migrated", str(args.migrated), "--migrated_sha256", args.migrated_sha256,
                    "--p33", str(args.p33), "--p33_sha256", args.p33_sha256,
                    "--i1", str(args.i1), "--i1_sha256", args.i1_sha256,
                    "--p34", str(args.p34), "--p34_sha256", args.p34_sha256,
                    "--p35", str(checkpoint), "--p35_sha256", checkpoint_sha,
                    "--control_reader", control_reader,
                    "--gate_ceiling_override", str(ceiling),
                ]
                if not summary_path.is_file():
                    subprocess.run(command, check=True)
                rows = read_jsonl(rows_path)
                if len(rows) != 1024:
                    raise RuntimeError("P3.5 score-bundle condition is incomplete")
                row_sets[label] = rows
                conditions[label] = {
                    "rows_path": str(rows_path),
                    "rows_sha256": sha256_file(rows_path),
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "paired": paired_read(rows),
                }

    primary_labels = [
        f"step_{entry['step']}_ema_ceiling_{P35_PRIMARY_EVAL_CEILING:.2f}"
        for entry in run["history"]
    ]
    adjacent = [
        {
            "left": left,
            "right": right,
            **churn(row_sets[left], row_sets[right]),
        }
        for left, right in zip(primary_labels, primary_labels[1:])
    ]
    final_step = run["history"][-1]["step"]
    raw_final = f"step_{final_step}_raw_ceiling_{P35_PRIMARY_EVAL_CEILING:.2f}"
    ema_final = f"step_{final_step}_ema_ceiling_{P35_PRIMARY_EVAL_CEILING:.2f}"
    output = {
        "kind": "paper2_phase3_p35_score_bundle_v1",
        "status": "complete_dev_only_no_training",
        "seed": run["seed"],
        "arm": run["arm"],
        "run_summary_sha256": sha256_file(args.run_summary),
        "conditions": conditions,
        "ema_primary_adjacent_churn": adjacent,
        "final_raw_vs_ema": churn(row_sets[raw_final], row_sets[ema_final]),
        "registered_primary": ema_final,
        "raw_after_decay_secondary": raw_final,
        "secondary_ceiling": P35_SECONDARY_EVAL_CEILING,
        "confirm_scored": False,
        "eval_e_scored": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
