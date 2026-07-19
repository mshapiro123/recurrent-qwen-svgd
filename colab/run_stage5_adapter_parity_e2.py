"""E2: outcome-only persistence continuation at the Arm E adapter budget."""

from __future__ import annotations

import hashlib
import os
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
    lora_eval_args,
    path_for_cli,
    read_json,
    restore_arm_e_checkpoint,
    run,
    sha256_file,
    write_json,
)
from colab.stage5_chain_consolidation_utils import backup_checkpoint_to_drive, publish_run
from training.adapter_parity_battery import score_e2_persistence


SOURCE_DATA = ROOT / "outputs/stage5/stage5_chain_anneal_20260703_160250/data"
TRAIN_DATA = SOURCE_DATA / "train_chain_symbol_sft.jsonl"
HELDOUT_DATA = SOURCE_DATA / "test_chain_mcq_heldout64.jsonl"
EXPECTED_DATA_SHA256 = {
    "train": "935780fe07592653c2065d6bd05bdec1edc8b9f837c7319e881bc85248d68c65",
    "heldout": "994ab7e4cb8b15a6ee3c3db8c950674a60af41f95a1f4e7f4a5e3b094aa9c500",
}


def canonical_text_sha256(path: str | Path) -> str:
    """Hash text with platform-independent LF newlines."""

    raw = Path(path).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _assert_data() -> dict[str, str]:
    observed = {
        "train": canonical_text_sha256(TRAIN_DATA),
        "heldout": canonical_text_sha256(HELDOUT_DATA),
    }
    if observed != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"E2 standard held-out data changed: {observed}")
    return observed


def _write_config(run_dir: Path, checkpoint: Path) -> Path:
    config: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "dtype": os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 512,
        "max_loops": 4,
        "loop_loss_mode": "annealed_chain_to_outcome",
        # A hold fraction of one makes the chain-label coefficient zero from step one.
        "chain_anneal_hold_frac": 1.0,
        "chain_outcome_loss_weight": 1.0,
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "minimum_effective_batch_size": 1,
        "seed": 0,
        "optimizer": "adamw",
        "reject_muon": True,
        "learning_rate": 1e-5,
        "adamw_lr": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": 1000,
        "save_every": 500,
        "log_every": 100,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "require_active_supervision": True,
        "require_nonzero_train_gradient": True,
        "output_dir": path_for_cli(run_dir / "train" / "outcome_only"),
        "resume_from": path_for_cli(checkpoint),
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": 4,
            "end_loop": 4,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "canary_every": 500,
        "canary_jsonl": path_for_cli(
            ROOT
            / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/data/base_capability_canary_64.jsonl"
        ),
        "canary_value_prefix": "name:",
        "canary_baseline_accuracy": 60 / 64,
        "canary_hard_stop_delta": -0.03,
        "synthetic_phase": "adapter_parity_e2",
        "synthetic_stage": "outcome_only_1000",
        **adapter_resume_config(),
    }
    path = run_dir / "config" / "outcome_only.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _eval(run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    out = run_dir / "eval"
    rows = out / "active_rows.jsonl"
    summary = out / "active_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(HELDOUT_DATA),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows),
            "--output_summary",
            path_for_cli(summary),
            "--loop_counts",
            "1,2,3,4",
            "--threshold",
            "0.71",
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            "letter:",
            "--progress_every",
            "64",
            *lora_eval_args(),
        ]
    )
    return read_json(summary)


def main() -> int:
    run_id = os.environ.get("STAGE5_ADAPTER_E2_RUN_ID", "stage5_adapter_parity_e2_20260719")
    run_dir = ROOT / "outputs/stage5" / run_id
    checkpoint, restore = restore_arm_e_checkpoint(run_dir / "restored" / "arm_e_final.pt")
    data_hashes = _assert_data()
    config = _write_config(run_dir, checkpoint)
    result = run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config),
            "--train_jsonl",
            path_for_cli(TRAIN_DATA),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        accepted_returncodes={0, 2},
    )
    train_dir = run_dir / "train" / "outcome_only"
    (train_dir / "train.log").write_text(result.stdout or "", encoding="utf-8")
    final_checkpoint = train_dir / "unfrozen_recurrent_step_1000.pt"
    if not final_checkpoint.exists():
        blocked = {
            "kind": "stage5_adapter_parity_e2",
            "run_id": run_id,
            "status": "blocked_training_did_not_reach_step_1000",
            "source_checkpoint_sha256": ARM_E_FINAL_SHA256,
            "restore_receipt": restore,
            "data_sha256": data_hashes,
            "train_config": path_for_cli(config),
            "training_summary": path_for_cli(
                train_dir / "train_unfrozen_recurrent_summary.json"
            ),
            "decision": {
                "verdict": "failed",
                "e4_authorized": False,
                "reason": "outcome-only continuation did not reach the registered endpoint",
            },
        }
        write_json(run_dir / "summary.json", blocked)
        publish_run(run_dir, message=f"Record blocked Arm E E2 persistence {run_id} [skip ci]")
        return 2
    training = read_json(train_dir / "train_unfrozen_recurrent_summary.json")
    assert_adapter_training_summary(training)
    evaluation = _eval(run_dir, final_checkpoint)
    active = evaluation["active_total"]
    above = evaluation["above_diagonal"]
    decision = score_e2_persistence(
        diagonal_correct=int(active["correct"]),
        diagonal_total=int(active["total"]),
        continue_count=int(above["iterate"]),
        hold_count=int(above["hold"]),
        above_total=int(above["n"]),
    )
    final_sha = sha256_file(final_checkpoint)
    backup = backup_checkpoint_to_drive(
        final_checkpoint,
        run_id=run_id,
        stage_name="outcome_only_final",
        enabled=True,
    )
    payload = {
        "kind": "stage5_adapter_parity_e2",
        "run_id": run_id,
        "status": "finished" if decision["e4_authorized"] else "blocked_e4_not_authorized",
        "source_checkpoint_sha256": ARM_E_FINAL_SHA256,
        "restore_receipt": restore,
        "data_sha256": data_hashes,
        "protocol": {
            "steps": 1000,
            "supervision": "outcome_only_chain_label_weight_zero",
            "optimizer": "adamw",
            "learning_rate": 1e-5,
            "bridge_prelude_lr_multiplier": 10.0,
            "reader": "active_full_symbols",
        },
        "train_config": path_for_cli(config),
        "training_summary": path_for_cli(train_dir / "train_unfrozen_recurrent_summary.json"),
        "final_checkpoint": path_for_cli(final_checkpoint),
        "final_checkpoint_sha256": final_sha,
        "final_checkpoint_drive_backup": backup,
        "evaluation_summary": path_for_cli(run_dir / "eval" / "active_summary.json"),
        "evaluation": {
            "active_diagonal": evaluation["active_diagonal"],
            "active_total": active,
            "above_diagonal": above,
        },
        "decision": decision,
        "full_block_reference": {
            "active_total": {"correct": 625, "total": 640, "accuracy": 625 / 640},
            "above_diagonal": {"continue": 357, "hold": 1, "total": 384},
        },
    }
    write_json(run_dir / "summary.json", payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Arm E E2 - {run_id}",
                "",
                f"- Verdict: `{decision['verdict']}`",
                f"- Active diagonal: `{active}`",
                f"- Above diagonal: `{above}`",
                f"- E4 authorized: `{decision['e4_authorized']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Arm E E2 persistence {run_id} [skip ci]")
    return 0 if decision["e4_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
