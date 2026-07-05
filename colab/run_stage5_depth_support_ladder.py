"""Train a depth-1..8 support ladder and score it on frozen depth-1..14 rows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import (
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


NONREGRESSION_FLOORS = {
    "1": 0.93,
    "2": 0.93,
    "3": 0.93,
    "4": 0.93,
    "5": 0.85,
    "6": 0.85,
    "7": 0.85,
    "8": 0.85,
}
LOCKED_ROWS_PER_DEPTH = 128
STRONG_SCALING_MIN_CORRECT = 91
ASYMPTOTE_REJECTION_MIN_CORRECT = 79
CHANCE_REJECTION_MIN_CORRECT = 14
ADJACENT_EXTENSION_MIN_CORRECT = 52


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


def depth_of(row: dict[str, Any]) -> int:
    return int(row.get("depth", row.get("synthetic_depth", 0)))


def rows_by_depth(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(depth_of(row))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def diagonal_counts(active_summary: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    matrix = active_summary.get("active_matrix", {})
    out: dict[str, dict[str, float | int]] = {}
    for depth, by_loop in matrix.items():
        cell = by_loop.get(depth, {}) if isinstance(by_loop, dict) else {}
        out[str(depth)] = {
            "correct": int(cell.get("correct", 0)),
            "total": int(cell.get("total", 0)),
            "accuracy": float(cell.get("accuracy", 0.0)),
        }
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def score_ladder(active_summary: dict[str, Any], *, rows_per_depth: int = LOCKED_ROWS_PER_DEPTH) -> dict[str, Any]:
    counts = diagonal_counts(active_summary)
    nonregression = {
        depth: {
            "accuracy": float(counts.get(depth, {}).get("accuracy", 0.0)),
            "floor": floor,
            "pass": float(counts.get(depth, {}).get("accuracy", 0.0)) >= floor,
        }
        for depth, floor in NONREGRESSION_FLOORS.items()
    }
    selected_correct = {
        depth: int(counts.get(str(depth), {}).get("correct", 0))
        for depth in range(9, 15)
    }
    adjacent_extension_pass = selected_correct[9] >= ADJACENT_EXTENSION_MIN_CORRECT
    asymptote_rejected = selected_correct[10] >= ASYMPTOTE_REJECTION_MIN_CORRECT
    scaling_pass = selected_correct[10] >= STRONG_SCALING_MIN_CORRECT and selected_correct[11] >= CHANCE_REJECTION_MIN_CORRECT
    strong_scaling_pass = (
        selected_correct[10] >= STRONG_SCALING_MIN_CORRECT
        and selected_correct[11] >= STRONG_SCALING_MIN_CORRECT
    )
    long_tail_above_chance = all(
        selected_correct[depth] >= CHANCE_REJECTION_MIN_CORRECT
        for depth in range(11, 15)
    )
    if strong_scaling_pass:
        verdict = "strong_scaling"
    elif scaling_pass:
        verdict = "scaling_with_depth11_above_chance"
    elif asymptote_rejected:
        verdict = "asymptote_rejected_at_depth10"
    elif adjacent_extension_pass:
        verdict = "adjacent_extension_only"
    else:
        verdict = "asymptote_or_failure"
    return {
        "diagonal_counts": counts,
        "locked_thresholds": {
            "rows_per_depth": int(rows_per_depth),
            "nonregression_floors": dict(NONREGRESSION_FLOORS),
            "adjacent_extension_depth9_min_correct": ADJACENT_EXTENSION_MIN_CORRECT,
            "depth10_asymptote_rejection_min_correct": ASYMPTOTE_REJECTION_MIN_CORRECT,
            "depth10_strong_scaling_min_correct": STRONG_SCALING_MIN_CORRECT,
            "depth11_strong_scaling_min_correct": STRONG_SCALING_MIN_CORRECT,
            "depth11_to_14_above_chance_min_correct": CHANCE_REJECTION_MIN_CORRECT,
        },
        "nonregression": nonregression,
        "nonregression_pass": all(item["pass"] for item in nonregression.values()),
        "selected_correct": selected_correct,
        "adjacent_extension_pass": adjacent_extension_pass,
        "asymptote_rejected": asymptote_rejected,
        "scaling_pass": scaling_pass,
        "selection_pass": scaling_pass,
        "overall_pass": all(item["pass"] for item in nonregression.values()) and scaling_pass,
        "strong_scaling_pass": strong_scaling_pass,
        "long_tail_above_chance": long_tail_above_chance,
        "verdict": verdict,
    }


def generate_depth_dataset(
    *,
    data_dir: Path,
    n_symbols: int,
    max_depth: int,
    rows_per_depth: int,
    seed: str,
    value_prefix: str,
) -> None:
    run(
        [
            sys.executable,
            "training/generate_synthetic_depth_task.py",
            "--output_dir",
            path_for_cli(data_dir),
            "--n_symbols",
            str(n_symbols),
            "--max_depth",
            str(max_depth),
            "--rows_per_depth",
            str(rows_per_depth),
            "--seed",
            seed,
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
            "--value_prefix",
            value_prefix,
        ]
    )


def ensure_base_frozen_eval_set(
    *,
    base_id: str,
    n_symbols: int,
    rows_per_depth: int,
    seed: str,
    value_prefix: str,
) -> dict[str, Any]:
    run_dir = ROOT / "outputs" / "stage5" / base_id
    summary_path = run_dir / "summary.json"
    data_dir = run_dir / "data"
    if summary_path.exists() and (data_dir / "test_chain_mcq.jsonl").exists() and (data_dir / "test_mcq.jsonl").exists():
        return read_json(summary_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    generate_depth_dataset(
        data_dir=data_dir,
        n_symbols=n_symbols,
        max_depth=10,
        rows_per_depth=rows_per_depth,
        seed=seed,
        value_prefix=value_prefix,
    )
    payload = {
        "kind": "stage5_synthetic_depth_frozen_eval_set",
        "run_id": base_id,
        "status": "finished",
        "n_symbols": n_symbols,
        "max_depth": 10,
        "rows_per_depth": rows_per_depth,
        "value_prefix": value_prefix,
        "seed": seed,
        "data_summary": path_for_cli(data_dir / "summary.json"),
        "test_chain_mcq": path_for_cli(data_dir / "test_chain_mcq.jsonl"),
        "test_mcq": path_for_cli(data_dir / "test_mcq.jsonl"),
    }
    write_json(summary_path, payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Frozen Synthetic Depth Eval Set - {base_id}",
                "",
                "- Depths: `1..10`",
                f"- Rows per depth: `{rows_per_depth}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Stage 5 frozen synthetic eval set {base_id} [skip ci]")
    return payload


def ensure_extended_frozen_eval_set(
    *,
    frozen_id: str,
    base_id: str,
    n_symbols: int,
    max_depth: int,
    rows_per_depth: int,
    seed: str,
    value_prefix: str,
) -> dict[str, Any]:
    run_dir = ROOT / "outputs" / "stage5" / frozen_id
    data_dir = run_dir / "data"
    summary_path = run_dir / "summary.json"
    test_chain = data_dir / "test_chain_mcq.jsonl"
    test_mcq = data_dir / "test_mcq.jsonl"
    if summary_path.exists() and test_chain.exists() and test_mcq.exists():
        return read_json(summary_path)

    base = ensure_base_frozen_eval_set(
        base_id=base_id,
        n_symbols=n_symbols,
        rows_per_depth=rows_per_depth,
        seed=seed,
        value_prefix=value_prefix,
    )
    tmp_dir = run_dir / "_generated_depth14"
    generate_depth_dataset(
        data_dir=tmp_dir,
        n_symbols=n_symbols,
        max_depth=max_depth,
        rows_per_depth=rows_per_depth,
        seed=seed,
        value_prefix=value_prefix,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    composed_files: dict[str, str] = {}
    depth_counts: dict[str, dict[str, int]] = {}
    for filename in ("test_chain_mcq.jsonl", "test_mcq.jsonl"):
        base_rows = [row for row in read_jsonl(base[filename.removesuffix(".jsonl")]) if depth_of(row) <= 10]
        extension_rows = [
            row
            for row in read_jsonl(tmp_dir / filename)
            if 10 < depth_of(row) <= max_depth
        ]
        rows = base_rows + extension_rows
        write_jsonl(data_dir / filename, rows)
        composed_files[filename.removesuffix(".jsonl")] = path_for_cli(data_dir / filename)
        depth_counts[filename.removesuffix(".jsonl")] = rows_by_depth(rows)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    payload = {
        "kind": "stage5_synthetic_depth_extended_frozen_eval_set",
        "run_id": frozen_id,
        "status": "finished",
        "base_frozen_eval_set": base,
        "extension_policy": "preserve depth 1..10 rows from base frozen v1; append generated depths 11..14",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "value_prefix": value_prefix,
        "seed": seed,
        "test_chain_mcq": composed_files["test_chain_mcq"],
        "test_mcq": composed_files["test_mcq"],
        "depth_counts": depth_counts,
    }
    write_json(summary_path, payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Extended Frozen Synthetic Depth Eval Set - {frozen_id}",
                "",
                "- Shared rows: depth `1..10` from `stage5_synthetic_depth_frozen_eval_v1` when available.",
                f"- Added rows: depth `11..{max_depth}` generated with seed `{seed}`.",
                f"- Rows per depth: `{rows_per_depth}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Stage 5 extended frozen synthetic eval set {frozen_id} [skip ci]")
    return payload


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    score = payload.get("ladder_score", {})
    lines = [
        f"# Depth Support Ladder - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Train support depth: `1..{payload.get('train_max_depth')}`",
        f"- Frozen eval depth: `1..{payload.get('eval_max_depth')}`",
        f"- Frozen eval set: `{payload.get('frozen_eval_set', {}).get('run_id')}`",
        f"- Verdict: `{score.get('verdict')}`",
        f"- Non-regression pass: `{score.get('nonregression_pass')}`",
        f"- Adjacent depth-9 extension pass: `{score.get('adjacent_extension_pass')}`",
        f"- Depth-10 asymptote rejected: `{score.get('asymptote_rejected')}`",
        f"- Scaling pass: `{score.get('scaling_pass')}`",
        f"- Strong scaling pass: `{score.get('strong_scaling_pass')}`",
        f"- Selected correct: `{score.get('selected_correct')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_LADDER_RUN_ID") or time.strftime("stage5_depth_support_ladder8_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_symbols = int(os.environ.get("STAGE5_LADDER_N_SYMBOLS", "16"))
    train_max_depth = int(os.environ.get("STAGE5_LADDER_TRAIN_MAX_DEPTH", "8"))
    eval_max_depth = int(os.environ.get("STAGE5_LADDER_EVAL_MAX_DEPTH", "14"))
    rows_per_depth = int(os.environ.get("STAGE5_LADDER_ROWS_PER_DEPTH", "256"))
    frozen_rows_per_depth = int(os.environ.get("STAGE5_LADDER_FROZEN_ROWS_PER_DEPTH", str(LOCKED_ROWS_PER_DEPTH)))
    total_steps = int(os.environ.get("STAGE5_LADDER_STEPS", "2000"))
    dtype = os.environ.get("STAGE5_LADDER_DTYPE", "bfloat16")
    threshold = float(os.environ.get("STAGE5_LADDER_THRESHOLD", "0.71"))
    value_prefix = os.environ.get("STAGE5_LADDER_VALUE_PREFIX", "letter:")
    frozen_id = os.environ.get("STAGE5_LADDER_FROZEN_EVAL_ID", "stage5_synthetic_depth_frozen_eval_v2_depth14")
    base_frozen_id = os.environ.get("STAGE5_LADDER_BASE_FROZEN_EVAL_ID", "stage5_synthetic_depth_frozen_eval_v1")
    frozen_seed = os.environ.get("STAGE5_LADDER_FROZEN_EVAL_SEED", "20260704")

    if train_max_depth != 8:
        raise ValueError("This registered ladder target expects STAGE5_LADDER_TRAIN_MAX_DEPTH=8.")
    if eval_max_depth < 14:
        raise ValueError("This registered ladder target expects eval coverage through at least depth 14.")

    frozen = ensure_extended_frozen_eval_set(
        frozen_id=frozen_id,
        base_id=base_frozen_id,
        n_symbols=n_symbols,
        max_depth=eval_max_depth,
        rows_per_depth=frozen_rows_per_depth,
        seed=frozen_seed,
        value_prefix=value_prefix,
    )

    payload: dict[str, Any] = {
        "kind": "stage5_depth_support_ladder",
        "run_id": run_id,
        "status": "started",
        "train_max_depth": train_max_depth,
        "eval_max_depth": eval_max_depth,
        "rows_per_depth": rows_per_depth,
        "frozen_rows_per_depth": frozen_rows_per_depth,
        "total_steps": total_steps,
        "threshold": threshold,
        "frozen_eval_set": frozen,
        "ladder_hypothesis": (
            "If the finite recurrent operator can be extended by support, training depths 1-8 should "
            "move the frozen depth-10/11 frontier beyond the support-6 asymptote."
        ),
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    run(
        [sys.executable, "colab/run_stage5_chain_anneal_to_outcome.py"],
        env={
            "STAGE5_ANNEAL_RUN_ID": run_id,
            "STAGE5_ANNEAL_LOOP_LOSS_MODE": "per_loop_labels",
            "STAGE5_ANNEAL_INIT_CHECKPOINT": os.environ.get(
                "STAGE5_LADDER_INIT_CHECKPOINT",
                "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            ),
            "STAGE5_ANNEAL_N_SYMBOLS": str(n_symbols),
            "STAGE5_ANNEAL_MAX_DEPTH": str(train_max_depth),
            "STAGE5_ANNEAL_ROWS_PER_DEPTH": str(rows_per_depth),
            "STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH": os.environ.get("STAGE5_LADDER_HELDOUT_ROWS_PER_DEPTH", "64"),
            "STAGE5_ANNEAL_TOTAL_STEPS": str(total_steps),
            "STAGE5_ANNEAL_HOLD_FRAC": os.environ.get("STAGE5_LADDER_SAVE_MID_FRAC", "0.5"),
            "STAGE5_ANNEAL_PRELUDE_LR_MULT": os.environ.get("STAGE5_LADDER_PRELUDE_LR_MULT", "10.0"),
            "STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                "STAGE5_LADDER_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
            ),
            "STAGE5_ANNEAL_DTYPE": dtype,
            "STAGE5_ANNEAL_VALUE_PREFIX": value_prefix,
        },
    )
    payload = read_json(run_dir / "summary.json")
    final_checkpoint = payload.get("final_checkpoint")
    if not final_checkpoint:
        raise RuntimeError("Ladder training did not produce final_checkpoint")

    artifact_summary = run_dir / "eval" / "frozen_depth14_artifact_check.json"
    active_rows = run_dir / "eval" / "frozen_depth14_active_rows.jsonl"
    active_summary = run_dir / "eval" / "frozen_depth14_active_summary.json"
    test_chain = ROOT / str(frozen["test_chain_mcq"])
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            path_for_cli(test_chain),
            "--checkpoint",
            str(final_checkpoint),
            "--output_summary",
            path_for_cli(artifact_summary),
            "--max_loops",
            str(eval_max_depth),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(test_chain),
            "--checkpoint",
            str(final_checkpoint),
            "--output_jsonl",
            path_for_cli(active_rows),
            "--output_summary",
            path_for_cli(active_summary),
            "--loop_counts",
            ",".join(str(idx) for idx in range(1, eval_max_depth + 1)),
            "--threshold",
            str(threshold),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    active_payload = read_json(active_summary)
    ladder_score = score_ladder(active_payload, rows_per_depth=frozen_rows_per_depth)
    payload.update(
        {
            "kind": "stage5_depth_support_ladder",
            "status": "finished_with_frozen_eval",
            "train_max_depth": train_max_depth,
            "eval_max_depth": eval_max_depth,
            "frozen_eval_set": frozen,
            "frozen_artifact_check": read_json(artifact_summary),
            "frozen_active_eval": {
                "active_rows": path_for_cli(active_rows),
                "active_summary": path_for_cli(active_summary),
                "active_diagonal": {
                    str(depth): float(value)
                    for depth, value in active_payload.get("active_diagonal", {}).items()
                },
                "active_total": active_payload.get("active_total", {}),
                "above_diagonal": active_payload.get("above_diagonal", {}),
            },
            "ladder_score": ladder_score,
            "decision_read": {
                "question": "Does training support through depth 8 move the active-label frontier past the support-6 asymptote?",
                "single_variable": "training_support_depth",
                "comparator": "stage5_depth_support_route_20260705_124320",
                "verdict": ladder_score["verdict"],
                "nonregression_pass": ladder_score["nonregression_pass"],
                "selected_correct": ladder_score["selected_correct"],
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 depth support ladder {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
