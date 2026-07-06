"""Continue the support-8 synthetic-depth ladder for an extra fixed-dose arm."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_depth_support_ladder import (
    locked_gate_summary,
    score_ladder,
    verify_constructive_synthetic_generator,
)
from colab.stage5_chain_consolidation_utils import (
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    resolve_checkpoint_reference,
    write_json,
)
from colab.stage5_support8_followup import (
    DEFAULT_SUPPORT8_SOURCE_SUMMARY,
    SUPPORT8_ROWS_PER_DEPTH,
    SUPPORT8_TRAIN_MAX_DEPTH,
    SUPPORT8_TRAIN_SEED,
    active_diagonal,
    decay_alignment,
    score_dose_arm,
    validate_support8_source_summary,
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


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    ladder = payload.get("ladder_score") or {}
    dose = payload.get("dose_score") or {}
    lines = [
        f"# Support-8 Dose Arm - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Extra steps: `{payload.get('extra_steps')}`",
        f"- Frozen eval set: `{payload.get('frozen_eval_set', {}).get('run_id')}`",
        f"- Ladder verdict: `{ladder.get('verdict')}`",
        f"- Dose verdict: `{dose.get('verdict')}`",
        f"- Scaling revived: `{dose.get('scaling_revived')}`",
        f"- Deceleration confirmed below 95: `{dose.get('deceleration_confirmed_below_95')}`",
        f"- Selected correct: `{ladder.get('selected_correct')}`",
        f"- Delta vs support-8 depth10/depth11: `{dose.get('depth10_delta_vs_support8')}`, `{dose.get('depth11_delta_vs_support8')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_DOSE_RUN_ID") or time.strftime("stage5_support8_dose_arm_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_summary = os.environ.get("STAGE5_DOSE_SOURCE_SUMMARY", DEFAULT_SUPPORT8_SOURCE_SUMMARY)
    source_payload = read_json(source_summary)
    source_check = validate_support8_source_summary(source_summary, source_payload)
    generator_preflight = verify_constructive_synthetic_generator()
    init_checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        source_summary,
        run_dir / "restored" / "support8_final.pt",
        label="support8_final",
    )
    frozen = source_payload.get("frozen_eval_set") or {}
    frozen_chain = frozen.get("test_chain_mcq")
    if not frozen_chain or not (ROOT / frozen_chain).exists():
        raise FileNotFoundError(f"Missing frozen depth-14 eval rows: {frozen_chain!r}")

    extra_steps = int(os.environ.get("STAGE5_DOSE_STEPS", "2000"))
    train_max_depth = int(os.environ.get("STAGE5_DOSE_MAX_DEPTH", str(SUPPORT8_TRAIN_MAX_DEPTH)))
    rows_per_depth = int(os.environ.get("STAGE5_DOSE_ROWS_PER_DEPTH", str(SUPPORT8_ROWS_PER_DEPTH)))
    if train_max_depth != SUPPORT8_TRAIN_MAX_DEPTH:
        raise ValueError("The registered dose arm expects STAGE5_DOSE_MAX_DEPTH=8.")
    if rows_per_depth != SUPPORT8_ROWS_PER_DEPTH:
        raise ValueError("The registered dose arm expects STAGE5_DOSE_ROWS_PER_DEPTH=256.")

    gates = locked_gate_summary(rows_per_depth=int(os.environ.get("STAGE5_DOSE_FROZEN_ROWS_PER_DEPTH", "128")))
    payload: dict[str, Any] = {
        "kind": "stage5_support8_dose_arm",
        "run_id": run_id,
        "status": "started",
        "source_summary": source_summary,
        "source_check": source_check,
        "source_decay_alignment": decay_alignment(active_diagonal(source_payload)),
        "extra_steps": extra_steps,
        "train_max_depth": train_max_depth,
        "rows_per_depth": rows_per_depth,
        "init_checkpoint": path_for_cli(init_checkpoint),
        "init_checkpoint_metadata": checkpoint_meta,
        "generator_preflight": generator_preflight,
        "locked_gates": gates,
        "frozen_eval_set": frozen,
        "dose_hypothesis": (
            "If the support-8 miss was dose-confounded, 2000 additional per-loop-label steps "
            "should revive locked depth-10 scaling or the soft depth-10/depth-11 frontier."
        ),
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    run(
        [sys.executable, "colab/run_stage5_chain_anneal_to_outcome.py"],
        env={
            "STAGE5_ANNEAL_RUN_ID": run_id,
            "STAGE5_ANNEAL_LOOP_LOSS_MODE": "per_loop_labels",
            "STAGE5_ANNEAL_INIT_CHECKPOINT": path_for_cli(init_checkpoint),
            "STAGE5_ANNEAL_N_SYMBOLS": os.environ.get("STAGE5_DOSE_N_SYMBOLS", "16"),
            "STAGE5_ANNEAL_MAX_DEPTH": str(train_max_depth),
            "STAGE5_ANNEAL_ROWS_PER_DEPTH": str(rows_per_depth),
            "STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH": os.environ.get("STAGE5_DOSE_HELDOUT_ROWS_PER_DEPTH", "64"),
            "STAGE5_ANNEAL_TOTAL_STEPS": str(extra_steps),
            "STAGE5_ANNEAL_HOLD_FRAC": os.environ.get("STAGE5_DOSE_SAVE_MID_FRAC", "0.5"),
            "STAGE5_ANNEAL_PRELUDE_LR_MULT": os.environ.get("STAGE5_DOSE_PRELUDE_LR_MULT", "10.0"),
            "STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                "STAGE5_DOSE_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
            ),
            "STAGE5_ANNEAL_DTYPE": os.environ.get("STAGE5_DOSE_DTYPE", "bfloat16"),
            "STAGE5_ANNEAL_VALUE_PREFIX": os.environ.get("STAGE5_DOSE_VALUE_PREFIX", "letter:"),
            "STAGE5_ANNEAL_SEED": os.environ.get("STAGE5_DOSE_SEED", SUPPORT8_TRAIN_SEED),
        },
    )
    payload = read_json(run_dir / "summary.json")
    final_checkpoint = payload.get("final_checkpoint")
    if not final_checkpoint:
        raise RuntimeError("Dose arm training did not produce final_checkpoint.")

    eval_max_depth = int(os.environ.get("STAGE5_DOSE_EVAL_MAX_DEPTH", "14"))
    threshold = float(os.environ.get("STAGE5_DOSE_THRESHOLD", "0.71"))
    dtype = os.environ.get("STAGE5_DOSE_DTYPE", "bfloat16")
    value_prefix = os.environ.get("STAGE5_DOSE_VALUE_PREFIX", "letter:")
    artifact_summary = run_dir / "eval" / "frozen_depth14_artifact_check.json"
    active_rows = run_dir / "eval" / "frozen_depth14_active_rows.jsonl"
    active_summary = run_dir / "eval" / "frozen_depth14_active_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            frozen_chain,
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
            frozen_chain,
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
    ladder_score = score_ladder(active_payload, rows_per_depth=int(gates["rows_per_depth"]))
    dose_score = score_dose_arm(ladder_score)
    payload.update(
        {
            "kind": "stage5_support8_dose_arm",
            "status": "finished_with_frozen_eval",
            "source_summary": source_summary,
            "source_check": source_check,
            "extra_steps": extra_steps,
            "train_max_depth": train_max_depth,
            "eval_max_depth": eval_max_depth,
            "generator_preflight": generator_preflight,
            "locked_gates": gates,
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
            "dose_score": dose_score,
            "decision_read": {
                "question": "Was the support-8 scaling miss dose-confounded?",
                "single_variable": "additional_training_steps_on_support8",
                "extra_steps": extra_steps,
                "verdict": dose_score["verdict"],
                "scaling_revived": dose_score["scaling_revived"],
                "deceleration_confirmed_below_95": dose_score["deceleration_confirmed_below_95"],
                "selected_correct": ladder_score["selected_correct"],
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 support-8 dose arm {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
