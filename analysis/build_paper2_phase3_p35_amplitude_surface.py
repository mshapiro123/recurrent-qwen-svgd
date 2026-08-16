"""Build the registered P3.5 Arm-S DEV amplitude-surface receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CEILINGS = (0.02, 0.05, 0.08, 0.11)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_name(seed: int, ceiling: float) -> str:
    return f"seed_{seed}_ceiling_{str(ceiling).replace('.', 'p')}"


def paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixes = [row for row in rows if not row["base_correct"] and row["augmented_correct"]]
    regressions = [row for row in rows if row["base_correct"] and not row["augmented_correct"]]
    groups = {}
    for group in sorted({str(row["panel_group"]) for row in rows}):
        selected = [row for row in rows if row["panel_group"] == group]
        group_fixes = sum(not row["base_correct"] and row["augmented_correct"] for row in selected)
        group_regressions = sum(row["base_correct"] and not row["augmented_correct"] for row in selected)
        groups[group] = {
            "rows": len(selected),
            "fixes": group_fixes,
            "regressions": group_regressions,
            "net_rows": group_fixes - group_regressions,
        }
    return {
        "rows": len(rows),
        "base_correct": sum(bool(row["base_correct"]) for row in rows),
        "augmented_correct": sum(bool(row["augmented_correct"]) for row in rows),
        "fixes": len(fixes),
        "regressions": len(regressions),
        "net_rows": len(fixes) - len(regressions),
        "by_panel_group": groups,
    }


def build(input_dir: Path, lock: dict[str, Any]) -> dict[str, Any]:
    conditions = {}
    panel_shas = set()
    eligible_by_seed: dict[int, dict[float, bool]] = {0: {}, 1: {}}
    for seed in (0, 1):
        for ceiling in CEILINGS:
            name = condition_name(seed, ceiling)
            rows_path = input_dir / f"{name}.jsonl"
            task_path = input_dir / f"{name}_summary.json"
            audit_path = input_dir / f"{name}_audit.json"
            rows = read_jsonl(rows_path)
            task = read_json(task_path)
            audit = read_json(audit_path)
            if len(rows) != 1024 or len({row["item_id"] for row in rows}) != 1024:
                raise RuntimeError(f"{name} task coverage changed")
            if any((task["confirm_scored"], task["eval_e_scored"], task["optimizer_constructed"])):
                raise RuntimeError(f"{name} violated evaluation-only scope")
            if int(task["optimizer_steps"]) != 0 or int(audit["optimizer_steps"]) != 0:
                raise RuntimeError(f"{name} constructed or stepped an optimizer")
            if float(task["evaluation_gate_ceiling"]) != ceiling or float(audit["ceiling"]) != ceiling:
                raise RuntimeError(f"{name} ceiling identity changed")
            panel_shas.add(str(task["panel_sha256"]))
            read = paired(rows)
            floor = read["by_panel_group"]["floor"]
            chi = float(audit["audit"]["collateral_chi"])
            eligible = chi == 0.0 and int(floor["net_rows"]) >= -2
            eligible_by_seed[seed][ceiling] = eligible
            conditions[name] = {
                **read,
                "seed": seed,
                "ceiling": ceiling,
                "collateral_chi": chi,
                "pi_dir": audit["audit"]["pi_dir"],
                "pi_dep": audit["audit"]["pi_dep"],
                "safety_eligible_this_seed": eligible,
                "task_rows_sha256": sha256_file(rows_path),
                "task_summary_sha256": sha256_file(task_path),
                "audit_summary_sha256": sha256_file(audit_path),
            }
    if len(panel_shas) != 1:
        raise RuntimeError("amplitude cells did not use one frozen DEV panel")
    replicated = {
        str(ceiling): bool(eligible_by_seed[0][ceiling] and eligible_by_seed[1][ceiling])
        for ceiling in CEILINGS
    }
    selected = max((ceiling for ceiling in CEILINGS if replicated[str(ceiling)]), default=None)
    return {
        "kind": "paper2_phase3_p35_amplitude_surface_v1",
        "status": "complete_dev_score_only",
        "authority": lock["authority"],
        "panel_sha256": next(iter(panel_shas)),
        "conditions": conditions,
        "replicated_safety_eligibility": replicated,
        "selected_ceiling_under_preregistered_rule": selected,
        "selection_rule": lock["amplitude_surface"]["selection_rule"],
        "previously_seen_ceilings": lock["amplitude_surface"]["previously_seen"],
        "scope": {
            "dev_only": True,
            "checkpoint_selection_barred": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.input_dir, read_json(args.lock))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
