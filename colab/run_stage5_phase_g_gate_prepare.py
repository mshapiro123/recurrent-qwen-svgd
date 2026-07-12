"""Prepare the no-GPU Phase G-alpha gate and lock its deterministic keeper."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.abductive_injective_task import (
    AbductiveInjectiveConfig,
    build_rows,
    validate_rows,
    write_jsonl,
)


DEFAULT_RECEIPT = ROOT / "outputs/stage5/stage5_natural_surface_receipts_20260709_210151/summary.json"
LOCKED_KEEPER_STEP = 2000
SYNTHETIC_GUARDRAIL_FLOOR = 0.93


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diagonal_min(payload: dict[str, Any], depths: range) -> float:
    diagonal = payload.get("active_diagonal") or {}
    values = [float(diagonal[str(depth)]) for depth in depths if str(depth) in diagonal]
    if len(values) != len(list(depths)):
        raise RuntimeError(f"active diagonal is incomplete for depths {depths.start}-{depths.stop - 1}")
    return min(values)


def keeper_gate_from_receipt(receipt: dict[str, Any], *, step: int = LOCKED_KEEPER_STEP) -> dict[str, Any]:
    step_key = f"step_{int(step)}"
    evals = receipt.get("evals") or {}
    if step_key not in evals:
        raise RuntimeError(f"receipt has no {step_key} evaluation")
    row = evals[step_key]
    relay = row["paired_relay_d1_12"]
    pointer = row["paired_pointer_d1_12"]
    synthetic = row["synthetic_frozen_v3_d1_12"]
    checkpoints = {relay.get("checkpoint"), pointer.get("checkpoint"), synthetic.get("checkpoint")}
    if None in checkpoints or len(checkpoints) != 1:
        raise RuntimeError(f"keeper evaluations do not share one checkpoint: {sorted(str(v) for v in checkpoints)}")
    checkpoint = str(next(iter(checkpoints)))

    backups = receipt.get("receipts", {}).get("drive_checkpoint_backup", {}).get("checkpoint_files", [])
    backup = next((item for item in backups if int(item.get("step", -1)) == int(step)), None)
    if not backup:
        raise RuntimeError(f"receipt has no backed-up checkpoint for step {step}")
    if str(backup.get("dest")) != checkpoint:
        raise RuntimeError("keeper checkpoint path does not match its durable backup receipt")

    synthetic_min = diagonal_min(synthetic, range(1, 13))
    result = {
        "source_run_id": receipt.get("source_run_id"),
        "step": int(step),
        "checkpoint": checkpoint,
        "checkpoint_sha256": backup.get("sha256"),
        "checkpoint_bytes": backup.get("bytes"),
        "synthetic_full_width_min_1_12": synthetic_min,
        "synthetic_guardrail_floor": SYNTHETIC_GUARDRAIL_FLOOR,
        "synthetic_guardrail_pass": synthetic_min >= SYNTHETIC_GUARDRAIL_FLOOR,
        "relay_min_1_8": diagonal_min(relay, range(1, 9)),
        "relay_min_9_12": diagonal_min(relay, range(9, 13)),
        "pointer_min_1_8": diagonal_min(pointer, range(1, 9)),
        "pointer_min_9_12": diagonal_min(pointer, range(9, 13)),
        "relay_total_accuracy": float(relay["active_total"]["accuracy"]),
        "pointer_total_accuracy": float(pointer["active_total"]["accuracy"]),
    }
    result["status"] = "green" if result["synthetic_guardrail_pass"] else "blocked"
    return result


def generate_gate_data(output_dir: Path, *, seed: int) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    all_ids: set[str] = set()
    for split, rows_per_depth in (("train", 256), ("test", 128)):
        for mode in ("injective", "abductive"):
            config = AbductiveInjectiveConfig(
                n_symbols=20,
                max_depth=8,
                rows_per_depth=rows_per_depth,
                seed=seed,
                min_solutions=2,
                max_solutions=4,
            )
            rows = build_rows(config, split=split, mode=mode)
            validation = validate_rows(rows, expected_mode=mode)
            if validation["status"] != "passed":
                raise RuntimeError(f"{split}/{mode} failed validation")
            ids = {str(row["id"]) for row in rows}
            if overlap := all_ids.intersection(ids):
                raise RuntimeError(f"dataset id overlap: {sorted(overlap)[:5]}")
            all_ids.update(ids)
            path = output_dir / "data" / f"{split}_{mode}.jsonl"
            write_jsonl(path, rows)
            datasets[f"{split}_{mode}"] = {"path": str(path.relative_to(ROOT)), **validation}
    return datasets


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    keeper = payload["keeper_gate"]
    lines = [
        f"# Phase G-alpha Gate Preparation - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Phase G-alpha ready: `{payload['phase_g_alpha_ready']}`",
        f"- Keeper: step `{keeper['step']}` at `{keeper['checkpoint']}`",
        f"- Keeper SHA256: `{keeper['checkpoint_sha256']}`",
        f"- Synthetic guardrail: `{keeper['synthetic_full_width_min_1_12']:.6f}` "
        f"(floor `{keeper['synthetic_guardrail_floor']:.2f}`)",
        f"- Relay min d1-8 / d9-12: `{keeper['relay_min_1_8']:.6f}` / `{keeper['relay_min_9_12']:.6f}`",
        f"- Pointer min d1-8 / d9-12: `{keeper['pointer_min_1_8']:.6f}` / `{keeper['pointer_min_9_12']:.6f}`",
        "",
        "## Data gate",
        "",
    ]
    for name, row in payload["datasets"].items():
        manifest = row["manifest"]
        lines.append(
            f"- `{name}`: `{row['status']}`, rows `{manifest['rows']}`, "
            f"solutions `{manifest['solution_counts']}`, sha `{manifest['row_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This opens data preparation for G-alpha. It does not establish a stochastic-width win. "
            "The next paid run is the deterministic task-learnability and answer-head-sampling baseline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=1_104_729)
    args = parser.parse_args()

    run_id = f"stage5_phase_g_gate_prepare_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs/stage5" / run_id
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    keeper = keeper_gate_from_receipt(read_json(args.receipt))
    datasets = generate_gate_data(output_dir, seed=args.seed)
    data_gate_pass = all(row["status"] == "passed" for row in datasets.values())
    payload = {
        "kind": "stage5_phase_g_gate_prepare",
        "run_id": output_dir.name,
        "status": "passed" if keeper["status"] == "green" and data_gate_pass else "blocked",
        "phase_g_alpha_ready": keeper["status"] == "green" and data_gate_pass,
        "receipt": str(Path(args.receipt)),
        "keeper_gate": keeper,
        "datasets": datasets,
        "locked_comparators": [
            "latent_K_vs_answer_head_temperature_at_matched_K",
            "latent_width_vs_deterministic_depth_at_iso_compute",
        ],
        "deferred_until_coverage_win": ["LPRM", "per_trajectory_halting", "SVGD"],
        "do_not_claim": (
            "This gate validates data and a deterministic keeper only; it provides no evidence that "
            "stochastic width improves coverage."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["phase_g_alpha_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
