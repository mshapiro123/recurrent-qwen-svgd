"""Run the parameter-matched distributed oracle-control localization probe."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab import run_stage5_phase_g_alpha as alpha  # noqa: E402
from colab.run_stage5_oracle_interface_probe import (  # noqa: E402
    CONTROL_JSONL,
    publish_receipts,
    read_json,
    run,
    sha256_file,
    train_arm,
    validate_source,
)
from training.oracle_intrablock_control_spec import (  # noqa: E402
    LOCKED_ROUTE,
    preregistration_payload,
)


CONTROL_RUN_DIR = (
    ROOT
    / "outputs"
    / "stage5"
    / "stage5_phase_g_oracle_interface_probe_20260718"
)
CONTROL_TRAIN_SUMMARY = CONTROL_RUN_DIR / "train" / "film" / "summary.json"
CONTROL_EVAL_SUMMARY = CONTROL_RUN_DIR / "eval" / "film" / "summary.json"


def assert_parameter_matched(
    layerwise: dict[str, Any],
    single_entry: dict[str, Any],
) -> None:
    layerwise_config = dict(layerwise["config"])
    single_entry_config = dict(single_entry["config"])
    matched_fields = (
        "keeper_sha256",
        "steps",
        "learning_rate",
        "weight_decay",
        "ema_decay",
        "bottleneck_dim",
        "seed",
        "max_length",
        "sampling_policy",
        "trainable_parameter_count",
    )
    mismatches = {
        field: (single_entry_config.get(field), layerwise_config.get(field))
        for field in matched_fields
        if single_entry_config.get(field) != layerwise_config.get(field)
    }
    if mismatches:
        raise AssertionError(
            "Layerwise probe is not parameter/training matched to historical FiLM: "
            f"{mismatches}"
        )


def main() -> int:
    run_id = os.environ.get(
        "STAGE5_ORACLE_INTRABLOCK_RUN_ID",
        "stage5_phase_g_oracle_intrablock_control_20260719",
    )
    steps = int(os.environ.get("STAGE5_ORACLE_INTRABLOCK_STEPS", "1500"))
    seed = int(os.environ.get("STAGE5_ORACLE_INTRABLOCK_SEED", "20260718"))
    bottleneck_dim = int(
        os.environ.get("STAGE5_ORACLE_INTRABLOCK_BOTTLENECK_DIM", "256")
    )
    dtype = os.environ.get("STAGE5_ORACLE_INTRABLOCK_DTYPE", "bfloat16")
    if steps != 1500 or seed != 20260718 or bottleneck_dim != 256:
        raise AssertionError(
            "Layerwise oracle steps, seed, and bottleneck are preregistered"
        )
    if not CONTROL_TRAIN_SUMMARY.exists() or not CONTROL_EVAL_SUMMARY.exists():
        raise FileNotFoundError(
            "Missing committed single-entry FiLM control receipts"
        )
    control_train = read_json(CONTROL_TRAIN_SUMMARY)
    control_eval = read_json(CONTROL_EVAL_SUMMARY)
    if control_train.get("route") != "film" or control_eval.get("route") != "film":
        raise AssertionError("Historical control receipts are not the FiLM arm")

    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    drive_checkpoint_dir = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / run_id
    )
    if drive_artifacts.exists():
        shutil.copytree(drive_artifacts, run_dir, dirs_exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    alpha.configure_runtime_transcript(run_dir / "runtime.log")
    validate_source()
    keeper = alpha.restore_keeper(run_dir)
    alpha.write_json(run_dir / "preregistration.json", preregistration_payload())
    summary: dict[str, Any] = {
        "kind": "stage5_phase_g_oracle_intrablock_control",
        "status": "started",
        "run_id": run_id,
        "keeper_sha256": alpha.KEEPER_SHA256,
        "route": LOCKED_ROUTE,
        "historical_control": str(CONTROL_EVAL_SUMMARY.relative_to(ROOT)),
        "steps": steps,
        "seed": seed,
        "bottleneck_dim": bottleneck_dim,
        "training_rows": 1899,
        "heldout_rows": 106,
        "heldout_groups": 32,
        "heldout_transitions": 305,
        "only_variable": "command_access_location",
        "variational_training_performed": False,
        "coverage_performed": False,
        "automatic_successor_authorized": False,
    }
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="startup",
        status="started",
        run_id=run_id,
    )
    publish_receipts(
        run_dir,
        f"Preregister Phase G distributed oracle probe {run_id} [skip ci]",
    )

    training = train_arm(
        route=LOCKED_ROUTE,
        keeper=keeper,
        run_dir=run_dir,
        drive_checkpoint_dir=drive_checkpoint_dir,
        drive_artifacts=drive_artifacts,
        steps=steps,
        seed=seed,
        bottleneck_dim=bottleneck_dim,
        dtype=dtype,
    )
    assert_parameter_matched(training, control_train)
    checkpoint = Path(training["ema_checkpoint"])
    checkpoint_sha = sha256_file(checkpoint)
    eval_dir = run_dir / "eval" / LOCKED_ROUTE
    cache_path = (
        drive_artifacts / "eval_cache" / LOCKED_ROUTE / "rows.jsonl"
    )
    run(
        [
            sys.executable,
            "eval/eval_oracle_interface_probe.py",
            "--data_jsonl",
            str(CONTROL_JSONL.relative_to(ROOT)),
            "--keeper",
            str(keeper),
            "--expected_keeper_sha256",
            alpha.KEEPER_SHA256,
            "--conditioner_checkpoint",
            str(checkpoint),
            "--expected_conditioner_sha256",
            checkpoint_sha,
            "--route",
            LOCKED_ROUTE,
            "--output_dir",
            str(eval_dir.relative_to(ROOT)),
            "--resume_cache_path",
            str(cache_path),
            "--bottleneck_dim",
            str(bottleneck_dim),
            "--dtype",
            dtype,
            "--device",
            "cuda",
        ]
    )
    arm_summary = eval_dir / "summary.json"
    gate_json = run_dir / "gate.json"
    gate_md = run_dir / "gate.md"
    run(
        [
            sys.executable,
            "eval/score_oracle_intrablock_control.py",
            "--arm_summary",
            str(arm_summary.relative_to(ROOT)),
            "--single_entry_control_summary",
            str(CONTROL_EVAL_SUMMARY.relative_to(ROOT)),
            "--output_json",
            str(gate_json.relative_to(ROOT)),
            "--output_md",
            str(gate_md.relative_to(ROOT)),
        ]
    )
    gate = read_json(gate_json)
    summary.update(
        {
            "status": "finished_terminal_localization",
            "training_summary": str(
                (run_dir / "train" / LOCKED_ROUTE / "summary.json").relative_to(
                    ROOT
                )
            ),
            "checkpoint_sha256": checkpoint_sha,
            "eval_summary": str(arm_summary.relative_to(ROOT)),
            "gate": gate,
        }
    )
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="terminal_gate",
        status="finished_terminal_localization",
        measured_reading=gate["measured_reading"],
        automatic_successor_authorized=False,
    )
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(
        run_dir,
        f"Record Phase G distributed oracle probe {run_id} [skip ci]",
    )
    return 0


def guarded_main() -> int:
    run_id = os.environ.get(
        "STAGE5_ORACLE_INTRABLOCK_RUN_ID",
        "stage5_phase_g_oracle_intrablock_control_20260719",
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    try:
        return main()
    except BaseException as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        alpha.record_runtime_failure(run_dir, drive_artifacts, exc)
        raise
    finally:
        alpha.configure_runtime_transcript(None)


if __name__ == "__main__":
    raise SystemExit(guarded_main())
