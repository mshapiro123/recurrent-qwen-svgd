"""E4: inverse-table acquisition and retention at the Arm E adapter budget."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.adapter_parity_common import (
    ARM_E_FINAL_SHA256,
    MODEL_NAME,
    ROOT,
    adapter_resume_config,
    assert_adapter_training_summary,
    path_for_cli,
    read_json,
    restore_arm_e_checkpoint,
    run,
    sha256_file,
    write_json,
)
from colab.run_stage5_inverse_composition_staircase import build_matched_data, _prepare_guardrail_data
from colab.run_stage5_inverse_table_rehearsal import (
    REHEARSAL_SOURCE,
    build_rehearsal_mix,
    fixed_schedule_dose,
    read_jsonl,
    rehearsal_optimizer_steps,
    rehearsal_weight_profiles,
    validate_causal_training_rows,
    write_jsonl,
)
from colab.stage5_chain_consolidation_utils import backup_checkpoint_to_drive, publish_run
from training.adapter_parity_battery import (
    score_e4_checkpoint_series,
    validate_e4_source,
    validate_e4_tier1_launch,
)


E2_SUMMARY_DEFAULT = ROOT / "outputs/stage5/stage5_adapter_parity_e2_20260719/summary.json"
ARM_E_CANARY = (
    ROOT
    / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/data/base_capability_canary_64.jsonl"
)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Could not parse checkpoint step: {path}")
    return int(match.group(1))


def _diagonal(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    data_jsonl: Path,
    max_depth: int,
    value_prefix: str,
) -> dict[str, Any]:
    out = run_dir / "guardrails" / label
    run(
        [
            sys.executable,
            "eval/eval_synthetic_diagonal_guardrail.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(out / "rows.jsonl"),
            "--output_summary",
            path_for_cli(out / "summary.json"),
            "--max_depth",
            str(max_depth),
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--lora_rank",
            "16",
            "--lora_alpha",
            "32",
            "--dtype",
            os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    return read_json(out / "summary.json")


def _inverse_eval(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    train_jsonl: Path,
    test_jsonl: Path,
) -> dict[str, Any]:
    out = run_dir / "eval" / label
    run(
        [
            sys.executable,
            "eval/eval_abductive_staircase.py",
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--test_jsonl",
            path_for_cli(test_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_dir",
            path_for_cli(out),
            "--rows_per_depth",
            "64",
            "--max_loops",
            "3",
            "--permutations",
            "0",
            "--bridge_projection_mode",
            "split",
            "--lora_rank",
            "16",
            "--lora_alpha",
            "32",
            "--dtype",
            os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--no-run_probes",
            "--no-include_train_predictions",
        ]
    )
    return read_json(out / "summary.json")


def _write_config(
    run_dir: Path,
    *,
    checkpoint: Path,
    train_jsonl: Path,
    max_steps: int,
    natural_canary: Path,
    natural_baseline_accuracy: float,
    tier1_canary: Path,
    tier1_baseline_accuracy: float,
) -> Path:
    cfg: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "dtype": os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 640,
        "max_loops": 12,
        "loop_loss_mode": "weighted_per_loop_labels",
        "row_specific_forward_loops": True,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "minimum_effective_batch_size": 8,
        "seed": 81_903,
        "optimizer": "adamw",
        "reject_muon": True,
        "learning_rate": 1e-5,
        "adamw_lr": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": int(max_steps),
        "save_every": 100,
        "log_every": 25,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "require_active_supervision": True,
        "require_nonzero_train_gradient": True,
        "output_dir": path_for_cli(run_dir / "train" / "inverse_rehearsal"),
        "resume_from": path_for_cli(checkpoint),
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": 12,
            "end_loop": 12,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        # Both registered retention guardrails are enforced on the live trajectory.
        "canary_every": 100,
        "canary_specs": [
            {
                "name": "natural_surface",
                "data_jsonl": path_for_cli(natural_canary),
                "baseline_accuracy": float(natural_baseline_accuracy),
                "hard_stop_delta": -0.03,
                "value_prefix": "name:",
                "mode": "diagonal",
                "max_depth": 8,
            },
            {
                "name": "tier1_arithmetic",
                "data_jsonl": path_for_cli(tier1_canary),
                "baseline_accuracy": float(tier1_baseline_accuracy),
                "hard_stop_delta": -0.03,
                "value_prefix": "name:",
                "mode": "loop1",
                "max_depth": 1,
            },
        ],
        "checkpoint_backup_every": 100,
        "checkpoint_backup_dir": str(
            Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints")
            / run_dir.name
            / "inverse_rehearsal"
        ),
        "synthetic_phase": "adapter_parity_e4",
        "synthetic_stage": "inverse_table_25pct_forward_rehearsal",
        **adapter_resume_config(),
    }
    path = run_dir / "config" / "inverse_rehearsal.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    e2_summary_path = Path(
        os.environ.get("STAGE5_ADAPTER_E4_E2_SUMMARY", str(E2_SUMMARY_DEFAULT))
    )
    if not e2_summary_path.is_absolute():
        e2_summary_path = ROOT / e2_summary_path
    e2 = read_json(e2_summary_path)
    authorization = validate_e4_source(e2)

    run_id = os.environ.get("STAGE5_ADAPTER_E4_RUN_ID", "stage5_adapter_parity_e4_20260719")
    run_dir = ROOT / "outputs/stage5" / run_id
    checkpoint, restore = restore_arm_e_checkpoint(run_dir / "restored" / "arm_e_final.pt")
    guardrails = _prepare_guardrail_data(run_dir)

    # Establish this arm's own baselines before any inverse training.
    natural_baseline = _diagonal(
        run_dir,
        label="arm_e_pretrain_natural",
        checkpoint=checkpoint,
        data_jsonl=guardrails["natural"],
        max_depth=8,
        value_prefix="name:",
    )
    tier1_baseline = _diagonal(
        run_dir,
        label="arm_e_pretrain_tier1",
        checkpoint=checkpoint,
        data_jsonl=ARM_E_CANARY,
        max_depth=1,
        value_prefix="name:",
    )
    tier1_total = {
        "correct": int(tier1_baseline.get("correct", -1)),
        "total": int(tier1_baseline.get("rows", -1)),
    }
    tier1_launch_gate = validate_e4_tier1_launch(**tier1_total)

    data_dir = run_dir / "data"
    matched = build_matched_data(data_dir)
    task_rows = [
        row
        for row in read_jsonl(data_dir / "train_C_inverse_table.jsonl")
        if int(row["depth"]) <= 3
    ]
    test_jsonl = data_dir / "test_C_inverse_table.jsonl"
    rehearsal_rows = read_jsonl(REHEARSAL_SOURCE)
    validate_causal_training_rows(task_rows, label="inverse-table task")
    validate_causal_training_rows(rehearsal_rows, label="forward rehearsal")
    max_steps = rehearsal_optimizer_steps(
        baseline_steps=250,
        effective_batch_size=8,
        rehearsal_fraction=0.25,
    )
    mixed, mix = build_rehearsal_mix(
        task_rows,
        rehearsal_rows,
        optimizer_steps=max_steps,
        effective_batch_size=8,
        rehearsal_fraction=0.25,
        seed=81_903,
    )
    profiles = rehearsal_weight_profiles(
        task_weights=[0.35294117647058826, 0.5294117647058824, 2.1176470588235294],
        rehearsal_depths=[
            int(row["depth"])
            for row in mixed
            if row["training_source"] == "forward_synthetic_rehearsal"
        ],
        max_loops=12,
    )
    for row in mixed:
        if row["training_source"] == "inverse_table_task":
            row["loop_label_weights"] = profiles["task"]
            row["forward_loop_count"] = 3
        else:
            row["loop_label_weights"] = profiles["rehearsal"]
            row["forward_loop_count"] = int(row["depth"])
    validate_causal_training_rows(mixed, label="Arm E mixed rehearsal")
    mix["dose"] = fixed_schedule_dose(mixed, max_loops=12)
    train_jsonl = data_dir / "train_inverse_plus_forward_rehearsal.jsonl"
    write_jsonl(train_jsonl, mixed)

    started = {
        "kind": "stage5_adapter_parity_e4",
        "run_id": run_id,
        "status": "baselines_ready",
        "source_checkpoint_sha256": ARM_E_FINAL_SHA256,
        "e2_authorization": authorization,
        "e2_summary": path_for_cli(e2_summary_path),
        "restore_receipt": restore,
        "matched_data": matched,
        "mix": mix,
        "natural_baseline": natural_baseline,
        "tier1_baseline": tier1_baseline,
        "tier1_launch_gate": tier1_launch_gate,
    }
    write_json(run_dir / "summary.json", started)
    publish_run(run_dir, message=f"Record Arm E E4 baselines {run_id} [skip ci]")

    config = _write_config(
        run_dir,
        checkpoint=checkpoint,
        train_jsonl=train_jsonl,
        max_steps=max_steps,
        natural_canary=guardrails["natural"],
        natural_baseline_accuracy=float(natural_baseline["accuracy"]),
        tier1_canary=ARM_E_CANARY,
        tier1_baseline_accuracy=60 / 64,
    )
    result = run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config),
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        accepted_returncodes={0, 2},
    )
    train_dir = run_dir / "train" / "inverse_rehearsal"
    (train_dir / "train.log").write_text(result.stdout or "", encoding="utf-8")
    training = read_json(train_dir / "train_unfrozen_recurrent_summary.json")
    assert_adapter_training_summary(training)
    checkpoints = sorted(train_dir.glob("unfrozen_recurrent_step_*.pt"), key=_checkpoint_step)
    if not checkpoints:
        started.update(
            {
                "status": "blocked_training_produced_no_checkpoint",
                "training_summary": path_for_cli(
                    train_dir / "train_unfrozen_recurrent_summary.json"
                ),
                "decision": {
                    "verdict": "wall_holds",
                    "joint_pass_any_checkpoint": False,
                    "reason": "training stopped before the first registered checkpoint",
                },
            }
        )
        write_json(run_dir / "summary.json", started)
        publish_run(run_dir, message=f"Record blocked Arm E E4 {run_id} [skip ci]")
        return 2

    baseline_natural_accuracy = float(natural_baseline["accuracy"])
    checkpoint_rows: list[dict[str, Any]] = []
    for candidate in checkpoints:
        step = _checkpoint_step(candidate)
        inverse = _inverse_eval(
            run_dir,
            label=f"step_{step}_inverse",
            checkpoint=candidate,
            train_jsonl=train_jsonl,
            test_jsonl=test_jsonl,
        )
        inverse_row = inverse["test"]["diagonal_by_depth"]["3"]
        synthetic = _diagonal(
            run_dir,
            label=f"step_{step}_synthetic",
            checkpoint=candidate,
            data_jsonl=guardrails["synthetic"],
            max_depth=12,
            value_prefix="letter:",
        )
        natural = _diagonal(
            run_dir,
            label=f"step_{step}_natural",
            checkpoint=candidate,
            data_jsonl=guardrails["natural"],
            max_depth=8,
            value_prefix="name:",
        )
        tier1 = _diagonal(
            run_dir,
            label=f"step_{step}_tier1",
            checkpoint=candidate,
            data_jsonl=ARM_E_CANARY,
            max_depth=1,
            value_prefix="name:",
        )
        checkpoint_rows.append(
            {
                "step": step,
                "checkpoint": path_for_cli(candidate),
                "checkpoint_sha256": sha256_file(candidate),
                "inverse_correct": int(inverse_row["correct"]),
                "inverse_total": int(inverse_row["total"]),
                "synthetic_min": float(synthetic["active_diagonal_min"]),
                "natural_baseline": baseline_natural_accuracy,
                "natural_accuracy": float(natural["accuracy"]),
                "tier1_correct": int(tier1["correct"]),
                "tier1_total": int(tier1["rows"]),
            }
        )
        started["status"] = f"evaluated_step_{step}"
        started["checkpoint_readouts"] = checkpoint_rows
        write_json(run_dir / "summary.json", started)
        publish_run(run_dir, message=f"Record Arm E E4 step {step} {run_id} [skip ci]")

    decision = score_e4_checkpoint_series(checkpoint_rows)
    final_checkpoint = checkpoints[-1]
    backup = backup_checkpoint_to_drive(
        final_checkpoint,
        run_id=run_id,
        stage_name="inverse_rehearsal_final",
        enabled=True,
    )
    started.update(
        {
            "status": "finished",
            "train_config": path_for_cli(config),
            "training_summary": path_for_cli(train_dir / "train_unfrozen_recurrent_summary.json"),
            "final_checkpoint": path_for_cli(final_checkpoint),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "final_checkpoint_drive_backup": backup,
            "checkpoint_readouts": checkpoint_rows,
            "decision": decision,
        }
    )
    write_json(run_dir / "summary.json", started)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Arm E E4 - {run_id}",
                "",
                f"- Verdict: `{decision['verdict']}`",
                f"- Any joint pass: `{decision['joint_pass_any_checkpoint']}`",
                f"- All retention checkpoints green: `{decision['all_retention_checkpoints_green']}`",
                f"- Final joint pass: `{decision['final_joint_pass']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Arm E E4 retention {run_id} [skip ci]")
    return 0 if decision["verdict"] == "wall_vanishes" else 2


if __name__ == "__main__":
    raise SystemExit(main())
