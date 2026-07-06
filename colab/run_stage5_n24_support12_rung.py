"""Run the final N-24, support-12 synthetic-depth rung.

This is the last synthetic rung by policy.  It trains support depths 1..12 on
N=24 symbols with planned dose, then scores frozen depths 1..22 with locked
gates.  The runner records the Tier-1 canary policy and can optionally run
external canary summaries, but it does not silently continue a failed canary
when one is provided.
"""

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

from colab.run_stage5_depth_support_ladder import generate_depth_dataset, verify_constructive_synthetic_generator
from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json
from colab.stage5_n24_rung import (
    N24_CHECKPOINTS,
    N24_MAX_EVAL_DEPTH,
    N24_ROWS_PER_DEPTH,
    N24_SUPPORT_DEPTH,
    N24_SYMBOLS,
    N24_TOTAL_STEPS,
    N24_TRAIN_ROWS_PER_DEPTH,
    locked_gate_summary,
    score_n24_rung,
    tier1_canary_verdict,
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


def parse_checkpoints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def ensure_frozen_eval_set(run_id: str, *, seed: str, value_prefix: str) -> dict[str, Any]:
    run_dir = ROOT / "outputs" / "stage5" / run_id
    summary_path = run_dir / "summary.json"
    data_dir = run_dir / "data"
    if summary_path.exists() and (data_dir / "test_chain_mcq.jsonl").exists():
        payload = read_json(summary_path)
        if int(payload.get("n_symbols", 0)) != N24_SYMBOLS or int(payload.get("max_depth", 0)) != N24_MAX_EVAL_DEPTH:
            raise RuntimeError(f"Existing frozen eval set has wrong shape: {summary_path}")
        return payload

    run_dir.mkdir(parents=True, exist_ok=True)
    generate_depth_dataset(
        data_dir=data_dir,
        n_symbols=N24_SYMBOLS,
        max_depth=N24_MAX_EVAL_DEPTH,
        rows_per_depth=N24_ROWS_PER_DEPTH,
        seed=seed,
        value_prefix=value_prefix,
    )
    payload = {
        "kind": "stage5_synthetic_depth_frozen_eval_set",
        "run_id": run_id,
        "status": "finished",
        "extension_generator": "constructive_distinct_orbit_prefix_no_rejection",
        "n_symbols": N24_SYMBOLS,
        "max_depth": N24_MAX_EVAL_DEPTH,
        "rows_per_depth": N24_ROWS_PER_DEPTH,
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
                f"# Frozen Synthetic Depth Eval Set - {run_id}",
                "",
                f"- Symbols: `{N24_SYMBOLS}`",
                f"- Depths: `1..{N24_MAX_EVAL_DEPTH}`",
                f"- Rows per depth: `{N24_ROWS_PER_DEPTH}`",
                "- Generator: `constructive_distinct_orbit_prefix_no_rejection`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Stage 5 N-24 frozen synthetic eval set {run_id} [skip ci]")
    return payload


def checkpoint_path(run_dir: Path, step: int) -> Path:
    path = run_dir / "train" / "chain_continuation" / f"unfrozen_recurrent_step_{step}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def eval_checkpoint(
    *,
    run_dir: Path,
    frozen_chain: str,
    checkpoint: Path,
    step: int,
    dtype: str,
    value_prefix: str,
    threshold: float,
) -> dict[str, Any]:
    eval_dir = run_dir / "eval" / f"frozen_depth22_step_{step}"
    artifact = eval_dir / "artifact_check.json"
    rows = eval_dir / "active_rows.jsonl"
    summary = eval_dir / "active_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            frozen_chain,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(artifact),
            "--max_loops",
            str(N24_MAX_EVAL_DEPTH),
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
            frozen_chain,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows),
            "--output_summary",
            path_for_cli(summary),
            "--loop_counts",
            ",".join(str(depth) for depth in range(1, N24_MAX_EVAL_DEPTH + 1)),
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
    active_payload = read_json(summary)
    return {
        "step": int(step),
        "checkpoint": path_for_cli(checkpoint),
        "artifact_check": read_json(artifact),
        "active_rows": path_for_cli(rows),
        "active_summary": path_for_cli(summary),
        "score": score_n24_rung(active_payload, rows_per_depth=N24_ROWS_PER_DEPTH),
    }


def canary_policy_from_env() -> dict[str, Any]:
    accuracy_delta = os.environ.get("STAGE5_RUNG_CANARY_ACCURACY_DELTA")
    ppl_delta = os.environ.get("STAGE5_RUNG_CANARY_PPL_RELATIVE_DELTA")
    verdict = tier1_canary_verdict(
        accuracy_delta=None if accuracy_delta in {None, ""} else float(accuracy_delta),
        ppl_relative_delta=None if ppl_delta in {None, ""} else float(ppl_delta),
    )
    return {
        "configured_every_steps": int(os.environ.get("STAGE5_RUNG_CANARY_EVERY", "1000")),
        "hard_stop_enabled": os.environ.get("STAGE5_RUNG_CANARY_HARD_STOP", "1") != "0",
        "provided_external_deltas": bool(accuracy_delta or ppl_delta),
        "verdict": verdict,
        "implementation_note": (
            "This runner enforces provided canary deltas before launch and records checkpoint-interval "
            "policy. True in-training 1000-step abort requires chunked training or a train-loop callback."
        ),
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    final = payload.get("checkpoint_evals", [])[-1] if payload.get("checkpoint_evals") else {}
    lines = [
        f"# N-24 Support-12 Final Synthetic Rung - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Frozen eval set: `{payload.get('frozen_eval_set', {}).get('run_id')}`",
        f"- Planned steps: `{payload.get('total_steps')}`",
        f"- Evaluated checkpoints: `{payload.get('evaluated_checkpoints')}`",
        f"- Final verdict: `{(final.get('score') or {}).get('verdict')}`",
        f"- Final selected correct: `{(final.get('score') or {}).get('selected_correct')}`",
        f"- Canary policy: `{payload.get('canary_policy', {}).get('verdict', {}).get('status')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_N24_RUN_ID") or time.strftime("stage5_n24_support12_rung_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = os.environ.get("STAGE5_N24_DTYPE", "bfloat16")
    value_prefix = os.environ.get("STAGE5_N24_VALUE_PREFIX", "letter:")
    threshold = float(os.environ.get("STAGE5_N24_THRESHOLD", "0.71"))
    eval_steps = parse_checkpoints(os.environ.get("STAGE5_N24_EVAL_CHECKPOINTS", ",".join(map(str, N24_CHECKPOINTS))))
    frozen_id = os.environ.get("STAGE5_N24_FROZEN_EVAL_ID", "stage5_synthetic_depth_frozen_eval_v3_depth22_n24")
    canary = canary_policy_from_env()
    if canary["hard_stop_enabled"] and canary["verdict"]["status"] == "red_hard_stop":
        raise RuntimeError(f"Tier-1 canary hard stop before N-24 launch: {canary}")

    generator_preflight = verify_constructive_synthetic_generator()
    frozen = ensure_frozen_eval_set(
        frozen_id,
        seed=os.environ.get("STAGE5_N24_FROZEN_EVAL_SEED", "20260706"),
        value_prefix=value_prefix,
    )
    payload: dict[str, Any] = {
        "kind": "stage5_n24_support12_rung",
        "run_id": run_id,
        "status": "started",
        "n_symbols": N24_SYMBOLS,
        "train_support_depth": N24_SUPPORT_DEPTH,
        "rows_per_depth": N24_TRAIN_ROWS_PER_DEPTH,
        "frozen_rows_per_depth": N24_ROWS_PER_DEPTH,
        "total_steps": int(os.environ.get("STAGE5_N24_STEPS", str(N24_TOTAL_STEPS))),
        "planned_checkpoints": list(N24_CHECKPOINTS),
        "evaluated_checkpoints": eval_steps,
        "generator_preflight": generator_preflight,
        "locked_gates": locked_gate_summary(rows_per_depth=N24_ROWS_PER_DEPTH),
        "canary_policy": canary,
        "frozen_eval_set": frozen,
        "stopping_rule": "Synthetic-depth line closes after this rung regardless of verdict.",
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    total_steps = int(payload["total_steps"])
    if total_steps != N24_TOTAL_STEPS:
        print(f"warning: non-registered STAGE5_N24_STEPS={total_steps}; locked plan is {N24_TOTAL_STEPS}", flush=True)
    run(
        [sys.executable, "colab/run_stage5_chain_anneal_to_outcome.py"],
        env={
            "STAGE5_ANNEAL_RUN_ID": run_id,
            "STAGE5_ANNEAL_LOOP_LOSS_MODE": "per_loop_labels",
            "STAGE5_ANNEAL_INIT_CHECKPOINT": os.environ.get(
                "STAGE5_N24_INIT_CHECKPOINT",
                "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            ),
            "STAGE5_ANNEAL_N_SYMBOLS": str(N24_SYMBOLS),
            "STAGE5_ANNEAL_MAX_DEPTH": str(N24_SUPPORT_DEPTH),
            "STAGE5_ANNEAL_ROWS_PER_DEPTH": str(N24_TRAIN_ROWS_PER_DEPTH),
            "STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH": os.environ.get("STAGE5_N24_HELDOUT_ROWS_PER_DEPTH", "64"),
            "STAGE5_ANNEAL_TOTAL_STEPS": str(total_steps),
            "STAGE5_ANNEAL_HOLD_FRAC": os.environ.get("STAGE5_N24_HOLD_FRAC", "0.6666666666666666"),
            "STAGE5_ANNEAL_PRELUDE_LR_MULT": os.environ.get("STAGE5_N24_PRELUDE_LR_MULT", "10.0"),
            "STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                "STAGE5_N24_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
            ),
            "STAGE5_ANNEAL_DTYPE": dtype,
            "STAGE5_ANNEAL_VALUE_PREFIX": value_prefix,
            "STAGE5_ANNEAL_SEED": os.environ.get("STAGE5_N24_TRAIN_SEED", "20260706"),
            "STAGE5_ANNEAL_LOG_EVERY": os.environ.get("STAGE5_N24_LOG_EVERY", "100"),
        },
    )

    payload = read_json(run_dir / "summary.json")
    checkpoint_evals = []
    frozen_chain = str(frozen["test_chain_mcq"])
    for step in eval_steps:
        checkpoint_evals.append(
            eval_checkpoint(
                run_dir=run_dir,
                frozen_chain=frozen_chain,
                checkpoint=checkpoint_path(run_dir, step),
                step=step,
                dtype=dtype,
                value_prefix=value_prefix,
                threshold=threshold,
            )
        )
        payload.update(
            {
                "kind": "stage5_n24_support12_rung",
                "status": "partial_eval_finished",
                "checkpoint_evals": checkpoint_evals,
            }
        )
        write_json(run_dir / "summary.json", payload)
        write_markdown(run_dir, payload)
        publish_run(run_dir, message=f"Record Stage 5 N-24 rung checkpoint {run_id} step {step} [skip ci]")

    payload.update(
        {
            "kind": "stage5_n24_support12_rung",
            "status": "finished_with_frozen_eval",
            "checkpoint_evals": checkpoint_evals,
            "decision_read": {
                "question": "Does the final synthetic rung confirm a four-point frontier law or characterize a ceiling?",
                "final_step": checkpoint_evals[-1]["step"] if checkpoint_evals else None,
                "final_verdict": checkpoint_evals[-1]["score"]["verdict"] if checkpoint_evals else "not_evaluated",
                "synthetic_line_closed": True,
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 N-24 support-12 final rung {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
