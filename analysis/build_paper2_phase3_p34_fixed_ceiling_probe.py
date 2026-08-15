"""Build the paired P3.4 fixed-ceiling score-probe receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = {
    (0, 0.02): "seed_0_ceiling_0p02",
    (0, 0.08): "seed_0_ceiling_0p08",
    (1, 0.02): "seed_1_ceiling_0p02",
    (1, 0.08): "seed_1_ceiling_0p08",
}
REGISTERED_ENDPOINT = {0: {"ceiling": 0.08, "correct": 507}, 1: {"ceiling": 0.02, "correct": 512}}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def paired_read(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixes = [row for row in rows if not row["base_correct"] and row["augmented_correct"]]
    regressions = [row for row in rows if row["base_correct"] and not row["augmented_correct"]]
    by_battery: dict[str, dict[str, int]] = {}
    for battery in sorted({str(row["battery"]) for row in rows}):
        selected = [row for row in rows if row["battery"] == battery]
        by_battery[battery] = {
            "rows": len(selected),
            "base_correct": sum(bool(row["base_correct"]) for row in selected),
            "augmented_correct": sum(bool(row["augmented_correct"]) for row in selected),
            "fixes": sum(not row["base_correct"] and row["augmented_correct"] for row in selected),
            "regressions": sum(row["base_correct"] and not row["augmented_correct"] for row in selected),
        }
    by_group: dict[str, dict[str, int]] = {}
    for group in sorted({str(row["panel_group"]) for row in rows}):
        selected = [row for row in rows if row["panel_group"] == group]
        by_group[group] = {
            "rows": len(selected),
            "base_correct": sum(bool(row["base_correct"]) for row in selected),
            "augmented_correct": sum(bool(row["augmented_correct"]) for row in selected),
            "fixes": sum(not row["base_correct"] and row["augmented_correct"] for row in selected),
            "regressions": sum(row["base_correct"] and not row["augmented_correct"] for row in selected),
        }
    return {
        "rows": len(rows),
        "base_correct": sum(bool(row["base_correct"]) for row in rows),
        "augmented_correct": sum(bool(row["augmented_correct"]) for row in rows),
        "net_rows": len(fixes) - len(regressions),
        "fixes": len(fixes),
        "regressions": len(regressions),
        "discordant_rows": len(fixes) + len(regressions),
        "by_battery": by_battery,
        "by_panel_group": by_group,
    }


def compare_ceilings(low: list[dict[str, Any]], high: list[dict[str, Any]]) -> dict[str, Any]:
    low_by_id = {str(row["item_id"]): row for row in low}
    high_by_id = {str(row["item_id"]): row for row in high}
    if set(low_by_id) != set(high_by_id):
        raise RuntimeError("fixed-ceiling row identities differ")
    transitions = Counter()
    by_battery: dict[str, Counter[str]] = defaultdict(Counter)
    for item_id, low_row in low_by_id.items():
        high_row = high_by_id[item_id]
        key = f"{int(bool(low_row['augmented_correct']))}_to_{int(bool(high_row['augmented_correct']))}"
        transitions[key] += 1
        by_battery[str(low_row["battery"])][key] += 1
    return {
        "low_to_high_transitions": dict(sorted(transitions.items())),
        "changed_rows": transitions["0_to_1"] + transitions["1_to_0"],
        "net_correct_change": transitions["0_to_1"] - transitions["1_to_0"],
        "by_battery": {name: dict(sorted(counts.items())) for name, counts in sorted(by_battery.items())},
    }


def build(input_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    paths: list[Path] = []
    conditions: dict[str, dict[str, Any]] = {}
    row_sets: dict[tuple[int, float], list[dict[str, Any]]] = {}
    panel_shas = set()
    for (seed, ceiling), name in CONDITIONS.items():
        row_path = input_dir / f"{name}.jsonl"
        summary_path = input_dir / f"{name}_summary.json"
        rows = read_jsonl(row_path)
        summary = read_json(summary_path)
        paths.extend([row_path, summary_path])
        if len(rows) != 1_024 or len({row["item_id"] for row in rows}) != 1_024:
            raise RuntimeError(f"{name} does not contain the fixed 1,024-row DEV panel")
        if int(summary["seed"]) != seed or float(summary["evaluation_gate_ceiling"]) != ceiling:
            raise RuntimeError(f"{name} summary identity changed")
        if summary["evaluation_gate_ceiling_source"] != "score_only_fixed_ceiling_override":
            raise RuntimeError(f"{name} did not use the score-only fixed-ceiling override")
        if summary["confirm_scored"] or summary["eval_e_scored"]:
            raise RuntimeError(f"{name} touched a sealed partition")
        if summary["optimizer_constructed"] or int(summary["optimizer_steps"]) != 0:
            raise RuntimeError(f"{name} is not evaluation-only")
        panel_shas.add(str(summary["panel_sha256"]))
        read = paired_read(rows)
        read["evaluation_gate_ceiling"] = ceiling
        read["row_receipt_sha256"] = sha256_file(row_path)
        read["condition_summary_sha256"] = sha256_file(summary_path)
        conditions[name] = read
        row_sets[(seed, ceiling)] = rows
    if len(panel_shas) != 1:
        raise RuntimeError("fixed-ceiling conditions used different DEV panels")

    reproductions = {}
    for seed, expected in REGISTERED_ENDPOINT.items():
        name = CONDITIONS[(seed, float(expected["ceiling"]))]
        observed = int(conditions[name]["augmented_correct"])
        if observed != int(expected["correct"]):
            raise RuntimeError(
                f"registered endpoint reproduction failed seed={seed}: expected={expected['correct']} observed={observed}"
            )
        reproductions[f"seed_{seed}"] = {
            "ceiling": expected["ceiling"],
            "expected_correct": expected["correct"],
            "observed_correct": observed,
            "exact": True,
        }

    comparisons = {
        f"seed_{seed}": compare_ceilings(row_sets[(seed, 0.02)], row_sets[(seed, 0.08)])
        for seed in (0, 1)
    }
    mean_net_by_ceiling = {}
    for ceiling in (0.02, 0.08):
        values = [conditions[CONDITIONS[(seed, ceiling)]]["net_rows"] for seed in (0, 1)]
        mean_net_by_ceiling[str(ceiling)] = sum(values) / len(values)
    seed1_target_floor = conditions[CONDITIONS[(1, 0.02)]]["by_panel_group"]
    return {
        "kind": "paper2_phase3_p34_fixed_ceiling_probe_v1",
        "status": "complete_score_only_dev",
        "panel_sha256": next(iter(panel_shas)),
        "conditions": conditions,
        "registered_endpoint_reproduction": reproductions,
        "paired_ceiling_comparison": comparisons,
        "mean_net_rows_by_ceiling": mean_net_by_ceiling,
        "seed_1_target_floor_reconstruction_at_registered_ceiling": seed1_target_floor,
        "scope": {
            "dev_only": True,
            "checkpoint_selection_barred": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "optimizer_steps": 0,
        },
    }, paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    args = parser.parse_args()
    summary, inputs = build(args.input_dir)
    write_json(args.output_summary, summary)
    write_json(args.output_manifest, {
        "kind": "paper2_phase3_p34_fixed_ceiling_probe_manifest_v1",
        "summary_sha256": sha256_file(args.output_summary),
        "inputs": {str(path): sha256_file(path) for path in inputs},
    })
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
