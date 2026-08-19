"""Assemble the registered Stage 2B step-5,000 adjudication receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.paper2_stage2b_depth import (
    amended_kill_gate_verdict,
    kill_gate_seed_read,
    kill_gate_trend_read,
)
from training.paper2_stage2b_runtime import atomic_json, sha256_file


REQUIRED_LOOKS = (3, 4, 5)
REQUIRED_TRANSITIONS = ("k2_to_k3", "k3_to_k4")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected an object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"expected row objects in {path}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty row receipt: {path}")
    return rows


def _ordered_margins(path: Path, *, seed: int, look: int) -> tuple[list[str], list[list[float]]]:
    rows = _read_jsonl(path)
    ordered = sorted(rows, key=lambda row: str(row["item_id"]))
    item_ids = [str(row["item_id"]) for row in ordered]
    if len(item_ids) != len(set(item_ids)):
        raise RuntimeError(f"duplicate DEV-2 item IDs in {path}")
    margins = []
    for row in ordered:
        if int(row["seed"]) != seed or int(row["look"]) != look:
            raise RuntimeError(f"seed/look mismatch in {path}")
        values = [float(value) for value in row["per_loop_mean_teacher_token_margin"]]
        if len(values) != 4:
            raise RuntimeError(f"expected four loop margins in {path}")
        margins.append(values)
    return item_ids, margins


def _transpose(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(row[index]) for row in rows] for index in range(4)]


def _transition_rows(rows: Sequence[Sequence[float]], transition: str) -> list[float]:
    index = REQUIRED_TRANSITIONS.index(transition) + 1
    return [float(row[index + 1]) - float(row[index]) for row in rows]


def adjudicate(seed_dirs: Mapping[int, Path]) -> dict[str, Any]:
    if set(seed_dirs) != {0, 1}:
        raise RuntimeError("Stage 2B adjudication requires seeds zero and one")

    seed_reads: dict[int, dict[str, Any]] = {}
    source_receipts: dict[str, Any] = {}
    trend_series: dict[int, dict[str, list[list[float]]]] = {}
    canonical_ids: list[str] | None = None
    watch_items: dict[str, Any] = {}

    for seed in (0, 1):
        directory = seed_dirs[seed]
        summary_path = directory.parent.parent / "receipts" / f"seed_{seed}" / "summary.json"
        if not summary_path.is_file():
            # The command also supports a directory containing both private rows and public receipts.
            summary_path = directory / "summary.json"
        summary = _read_json(summary_path)
        if int(summary.get("seed", -1)) != seed or int(summary.get("step", -1)) != 5000:
            raise RuntimeError(f"seed {seed} has not completed the registered step-5,000 stop")
        if summary.get("status") != "awaiting_step_5000_strategy_adjudication":
            raise RuntimeError(f"seed {seed} is not awaiting registered adjudication")
        if bool(summary.get("confirm_scored")) or bool(summary.get("eval_e_scored")):
            raise RuntimeError("sealed partition contact detected")

        trend_series[seed] = {transition: [] for transition in REQUIRED_TRANSITIONS}
        look_rows: dict[int, list[list[float]]] = {}
        look_ids: list[str] | None = None
        for look in REQUIRED_LOOKS:
            row_path = directory / f"dev2_margin_rows_look_{look}.jsonl"
            ids, rows = _ordered_margins(row_path, seed=seed, look=look)
            if look_ids is None:
                look_ids = ids
            elif ids != look_ids:
                raise RuntimeError(f"seed {seed} DEV-2 row pairing changed across looks")
            if canonical_ids is None:
                canonical_ids = ids
            elif ids != canonical_ids:
                raise RuntimeError("DEV-2 row pairing differs across seeds")
            look_rows[look] = rows
            for transition in REQUIRED_TRANSITIONS:
                trend_series[seed][transition].append(_transition_rows(rows, transition))
            source_receipts[f"seed_{seed}_look_{look}_rows"] = {
                "path": str(row_path),
                "sha256": sha256_file(row_path),
                "rows": len(rows),
            }

        seed_reads[seed] = kill_gate_seed_read(_transpose(look_rows[5]))
        look_path = directory.parent.parent / "receipts" / f"seed_{seed}" / "look_5.json"
        if not look_path.is_file():
            look_path = directory / "look_5.json"
        look_receipt = _read_json(look_path)
        if bool(look_receipt.get("confirm_scored")) or bool(look_receipt.get("eval_e_scored")):
            raise RuntimeError("sealed partition contact detected in look receipt")
        safety = look_receipt["dev1"]["safety"]
        if not bool(safety.get("pass")):
            raise RuntimeError(f"seed {seed} failed the DEV-1 hard floor")
        finite = look_receipt["finite_horizon"]
        if bool(finite.get("catastrophe")):
            raise RuntimeError(f"seed {seed} tripped the finite-horizon catastrophe rule")
        watch_items[str(seed)] = {
            "loop1_kl": look_receipt["loop1_kl"],
            "finite_horizon": finite,
            "r2_desk_read": look_receipt["r2_desk_read"],
            "dev1_safety": safety,
        }
        source_receipts[f"seed_{seed}_summary"] = {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        }
        source_receipts[f"seed_{seed}_look_5"] = {
            "path": str(look_path),
            "sha256": sha256_file(look_path),
        }

    trend_read = None
    if not any(bool(read["separating"]) for read in seed_reads.values()):
        trend_read = kill_gate_trend_read(trend_series)
    verdict = amended_kill_gate_verdict(
        seed_reads,
        step=5000,
        trend_read=trend_read,
    )
    return {
        "kind": "paper2_stage2b_step5000_adjudication_v1",
        "step": 5000,
        "seed_reads": {str(seed): read for seed, read in seed_reads.items()},
        "trend_read": trend_read,
        "verdict": verdict,
        "dev2_rows": len(canonical_ids or []),
        "watch_items": watch_items,
        "source_receipts": source_receipts,
        "confirm_scored": False,
        "eval_e_scored": False,
        "sealed_partitions_remain_sealed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-0-dir", type=Path, required=True)
    parser.add_argument("--seed-1-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = adjudicate({0: args.seed_0_dir, 1: args.seed_1_dir})
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
