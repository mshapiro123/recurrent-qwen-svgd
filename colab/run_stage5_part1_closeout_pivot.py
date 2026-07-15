"""Run the final deterministic micro-test and branching width-substrate screen."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import restore_checkpoint, run, sha256_file  # noqa: E402
from colab.run_stage5_phase_g_gate_prepare import DEFAULT_RECEIPT, keeper_gate_from_receipt  # noqa: E402
from colab.stage5_publish_utils import publishable_artifact_paths  # noqa: E402
from training.branching_relations_task import (  # noqa: E402
    BranchingRelationsConfig,
    build_rows as build_branching_rows,
    validate_rows as validate_branching_rows,
)
from training.continuation_policy import (  # noqa: E402
    GuardrailFloor,
    assert_launch_guardrail_floors,
    assert_training_lineage,
)
from training.loop_position_transfer_task import (  # noqa: E402
    LoopPositionConfig,
    build_eval_rows,
    build_training_rows,
    validate_loop_position_rows,
)


NATURAL_KEEPER_SHA256 = "0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f"
NATURAL_KEEPER_DRIVE = (
    "/content/drive/MyDrive/recurrent-qwen-svgd-backups/natural_surface_backup_20260709_180835/"
    "checkpoints/stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/"
    "unfrozen_recurrent_step_2000.pt"
)
N24_KEEPER_SHA256 = "898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc"
N24_KEEPER_DRIVE = (
    "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_n24_support12_rung_20260707_140139/"
    "anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt"
)
SYNTHETIC_GUARDRAIL_SOURCE = ROOT / (
    "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
)
TRAINED_POSITION_FLOOR = 0.71
SYNTHETIC_GUARDRAIL_FLOOR = 0.93


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return value.as_posix()


def _publish(run_dir: Path, message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        if "data" in path.relative_to(run_dir).parts:
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    pushed = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if pushed.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def _balanced_prefix(rows: list[dict[str, Any]], *, per_depth: int, max_depth: int) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for row in rows:
        depth = int(row["depth"])
        if depth <= max_depth and counts.get(depth, 0) < per_depth:
            selected.append(row)
            counts[depth] = counts.get(depth, 0) + 1
    expected = {depth: per_depth for depth in range(1, max_depth + 1)}
    if counts != expected:
        raise RuntimeError(f"Could not create balanced guardrail subset: {counts} != {expected}")
    return selected


def prepare_data(run_dir: Path) -> dict[str, Any]:
    data_dir = run_dir / "data"
    loop_cfg = LoopPositionConfig()
    datasets = {
        "loop_train": build_training_rows(loop_cfg),
        "loop_trained_eval": build_eval_rows(
            LoopPositionConfig(rows_per_position=64), prefix_lengths=(0, 1), split="trained_eval"
        ),
        "loop_transfer_eval": build_eval_rows(
            LoopPositionConfig(rows_per_position=128), prefix_lengths=(2, 3), split="transfer_eval"
        ),
        "branching_n20_verbal": build_branching_rows(
            BranchingRelationsConfig(), split="test", rendering="verbal", n_symbols=20
        ),
        "branching_n24_symbolic": build_branching_rows(
            BranchingRelationsConfig(), split="test", rendering="symbolic", n_symbols=24
        ),
    }
    manifests: dict[str, Any] = {}
    for name, rows in datasets.items():
        validation = (
            validate_branching_rows(rows)
            if name.startswith("branching")
            else validate_loop_position_rows(rows)
        )
        if validation["status"] != "passed":
            raise RuntimeError(f"Generated {name} rows failed validation: {validation['errors'][:5]}")
        write_jsonl(data_dir / f"{name}.jsonl", rows)
        manifests[name] = validation
    guardrail_rows = _balanced_prefix(read_jsonl(SYNTHETIC_GUARDRAIL_SOURCE), per_depth=16, max_depth=12)
    write_jsonl(data_dir / "synthetic_guardrail_d1_12_16_each.jsonl", guardrail_rows)
    manifests["synthetic_guardrail"] = {
        "rows": len(guardrail_rows),
        "depth_counts": {str(depth): 16 for depth in range(1, 13)},
    }
    write_json(run_dir / "data_manifest.json", manifests)
    return manifests


def _write_micro_config(path: Path, *, checkpoint: Path, output_dir: Path, seed: int) -> None:
    lineage = assert_training_lineage(
        regime="disposable_measurement",
        full_block_trainable=True,
        checkpoint_promotable=False,
        successor_source_allowed=False,
    )
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": os.environ.get("STAGE5_PART1_PIVOT_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 640,
        "max_loops": 2,
        "loop_loss_mode": "per_loop_labels",
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "minimum_effective_batch_size": 8,
        "seed": int(seed),
        "optimizer": "adamw",
        "learning_rate": 1e-5,
        "adamw_lr": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": 1000,
        "save_every": 500,
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
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(checkpoint),
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "train_auxiliary": {"bridge": True, "halting": False, "reentry_adapter": False, "latent": False},
        "recurrence_curriculum": {"enabled": False, "start_loop": 2, "end_loop": 2, "ramp_compute": False},
        "lineage_policy": lineage,
        "synthetic_phase": "loop_position_transfer_micro_test",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def _active_eval(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    data: Path,
    loop_counts: str,
) -> dict[str, Any]:
    output_dir = run_dir / "eval" / label
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(data),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(output_dir / "rows.jsonl"),
            "--output_summary",
            path_for_cli(output_dir / "summary.json"),
            "--loop_counts",
            loop_counts,
            "--threshold",
            str(TRAINED_POSITION_FLOOR),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            "name:",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_PART1_PIVOT_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    return read_json(output_dir / "summary.json")


def _guardrail_eval(run_dir: Path, *, label: str, checkpoint: Path) -> dict[str, Any]:
    output_dir = run_dir / "guardrails" / label
    run(
        [
            sys.executable,
            "eval/eval_synthetic_diagonal_guardrail.py",
            "--data_jsonl",
            path_for_cli(run_dir / "data" / "synthetic_guardrail_d1_12_16_each.jsonl"),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(output_dir / "rows.jsonl"),
            "--output_summary",
            path_for_cli(output_dir / "summary.json"),
            "--max_depth",
            "12",
            "--value_prefix",
            "letter:",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_PART1_PIVOT_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    summary = read_json(output_dir / "summary.json")
    return {
        "summary": path_for_cli(output_dir / "summary.json"),
        "active_diagonal_min": float(summary["active_diagonal_min"]),
        "floor": SYNTHETIC_GUARDRAIL_FLOOR,
        "passed": float(summary["active_diagonal_min"]) >= SYNTHETIC_GUARDRAIL_FLOOR,
    }


def _trained_position_gate(summary: dict[str, Any]) -> dict[str, Any]:
    diagonal = summary.get("active_diagonal") or {}
    values = {str(depth): float(diagonal.get(str(depth), 0.0)) for depth in (1, 2)}
    return {
        "accuracy_by_inverse_position": values,
        "floor": TRAINED_POSITION_FLOOR,
        "passed": all(value >= TRAINED_POSITION_FLOOR for value in values.values()),
    }


def interpret_loop_position(summary: dict[str, Any]) -> dict[str, Any]:
    diagonal = summary.get("active_diagonal") or {}
    values = [float(diagonal.get(str(depth), 0.0)) for depth in (3, 4)]
    minimum = min(values) if values else 0.0
    if minimum >= 0.55:
        reading = "substantially_position_invariant"
    elif max(values, default=0.0) <= 0.15:
        reading = "per_position_installation_confirmed"
    else:
        reading = "partial_transfer"
    return {"position_3_accuracy": values[0], "position_4_accuracy": values[1], "reading": reading}


def run_loop_position_micro_test(run_dir: Path, keeper: Path) -> dict[str, Any]:
    def delete_disposable_checkpoints() -> None:
        for checkpoint in run_dir.glob("micro_test/train/stage_*/*.pt"):
            checkpoint.unlink(missing_ok=True)

    candidates: list[dict[str, Any]] = []
    source = keeper
    for stage_index in (1, 2):
        train_dir = run_dir / "micro_test" / "train" / f"stage_{stage_index}"
        config = run_dir / "micro_test" / "configs" / f"stage_{stage_index}.yaml"
        _write_micro_config(config, checkpoint=source, output_dir=train_dir, seed=7_150_000 + stage_index)
        process = run(
            [
                sys.executable,
                "training/train_unfrozen_recurrent.py",
                "--config",
                path_for_cli(config),
                "--train_jsonl",
                path_for_cli(run_dir / "data" / "loop_train.jsonl"),
                "--device",
                os.environ.get("DEVICE", "cuda"),
            ]
        )
        (train_dir / "train.log").write_text(process.stdout or "", encoding="utf-8")
        checkpoints = sorted(train_dir.glob("unfrozen_recurrent_step_*.pt"), key=_checkpoint_step)
        for checkpoint in checkpoints:
            local_step = _checkpoint_step(checkpoint)
            cumulative_step = (stage_index - 1) * 1000 + local_step
            summary = _active_eval(
                run_dir,
                label=f"micro_trained_positions_step_{cumulative_step}",
                checkpoint=checkpoint,
                data=run_dir / "data" / "loop_trained_eval.jsonl",
                loop_counts="1,2",
            )
            candidates.append(
                {
                    "cumulative_step": cumulative_step,
                    "checkpoint": path_for_cli(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "trained_position_gate": _trained_position_gate(summary),
                }
            )
        stage_final = checkpoints[-1]
        guardrail = _guardrail_eval(
            run_dir,
            label=f"micro_stage_{stage_index}_step_{stage_index * 1000}",
            checkpoint=stage_final,
        )
        candidates[-1]["guardrail"] = guardrail
        if not guardrail["passed"]:
            delete_disposable_checkpoints()
            return {
                "status": "blocked_guardrail_hard_stop",
                "lineage_regime": "disposable_measurement",
                "candidates": candidates,
                "measurement": None,
                "disposable_checkpoints_deleted_after_measurement": True,
            }
        selected = next((row for row in candidates if row["trained_position_gate"]["passed"]), None)
        if selected is not None:
            selected_path = ROOT / selected["checkpoint"]
            transfer = _active_eval(
                run_dir,
                label=f"micro_transfer_step_{selected['cumulative_step']}",
                checkpoint=selected_path,
                data=run_dir / "data" / "loop_transfer_eval.jsonl",
                loop_counts="3,4",
            )
            result = {
                "status": "measurement_complete",
                "lineage_regime": "disposable_measurement",
                "checkpoint_promotable": False,
                "successor_source_allowed": False,
                "candidates": candidates,
                "selected": selected,
                "measurement": interpret_loop_position(transfer),
                "measurement_summary": path_for_cli(
                    run_dir / "eval" / f"micro_transfer_step_{selected['cumulative_step']}" / "summary.json"
                ),
            }
            delete_disposable_checkpoints()
            result["disposable_checkpoints_deleted_after_measurement"] = True
            return result
        source = stage_final
    delete_disposable_checkpoints()
    return {
        "status": "trained_position_prerequisite_failed",
        "lineage_regime": "disposable_measurement",
        "candidates": candidates,
        "measurement": None,
        "disposable_checkpoints_deleted_after_measurement": True,
    }


def run_branching_screen(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    data_name: str,
) -> dict[str, Any]:
    output_dir = run_dir / "branching_screen" / label
    run(
        [
            sys.executable,
            "eval/eval_branching_relations.py",
            "--data_jsonl",
            path_for_cli(run_dir / "data" / f"{data_name}.jsonl"),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(output_dir / "rows.jsonl"),
            "--output_summary",
            path_for_cli(output_dir / "summary.json"),
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_PART1_PIVOT_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    summary = read_json(output_dir / "summary.json")
    return {
        "summary": path_for_cli(output_dir / "summary.json"),
        "gate": summary["gate"],
        "by_reachable_set_stratum": summary["by_reachable_set_stratum"],
    }


def _write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    micro = payload.get("loop_position_micro_test") or {}
    lines = [
        f"# Part 1 Closeout Pivot Session - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Loop-position status: `{micro.get('status', 'pending')}`",
        f"- Loop-position reading: `{(micro.get('measurement') or {}).get('reading', 'not_measured')}`",
        f"- Phase G-alpha: `{payload.get('phase_g_alpha_status', 'closed')}`",
        "",
        "## Branching screens",
        "",
        "| Keeper | Pooled validity | Gate |",
        "|---|---:|---|",
    ]
    for name, screen in (payload.get("branching_screens") or {}).items():
        gate = screen.get("gate") or {}
        lines.append(f"| {name} | {gate.get('pooled_accuracy', 0.0):.4f} | {gate.get('passed', False)} |")
    lines.extend(
        [
            "",
            "The loop-position checkpoint is disposable and cannot be promoted or used as a source.",
            "A zero-shot double miss requires strategy review; adapter training is not auto-launched without a locked near-miss definition.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def main() -> int:
    run_id = os.environ.get("STAGE5_PART1_PIVOT_RUN_ID") or time.strftime(
        "stage5_part1_closeout_pivot_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": "stage5_part1_closeout_pivot_session",
        "run_id": run_id,
        "status": "started",
        "phase_g_alpha_status": "closed_pending_branching_screen",
    }
    keeper_gate = keeper_gate_from_receipt(read_json(DEFAULT_RECEIPT))
    floor_receipt = {"synthetic": {"min_accuracy": keeper_gate["synthetic_full_width_min_1_12"]}}
    payload["launch_floor_assertion"] = assert_launch_guardrail_floors(
        floor_receipt,
        [GuardrailFloor("synthetic.min_accuracy", SYNTHETIC_GUARDRAIL_FLOOR)],
    )
    natural, natural_restore = restore_checkpoint(
        [NATURAL_KEEPER_DRIVE, keeper_gate["checkpoint"]],
        run_dir / "restored" / "natural_step2000.pt",
        label="part1_pivot_natural_keeper",
    )
    if natural_restore["selected_checkpoint_sha256"] != NATURAL_KEEPER_SHA256:
        raise RuntimeError("Natural keeper SHA mismatch")
    n24, n24_restore = restore_checkpoint(
        [N24_KEEPER_DRIVE],
        run_dir / "restored" / "n24_step6000.pt",
        label="part1_pivot_n24_keeper",
    )
    if n24_restore["selected_checkpoint_sha256"] != N24_KEEPER_SHA256:
        raise RuntimeError("N24 keeper SHA mismatch")
    payload["keepers"] = {
        "natural_step2000": {**natural_restore, "expected_sha256": NATURAL_KEEPER_SHA256},
        "n24_step6000": {**n24_restore, "expected_sha256": N24_KEEPER_SHA256},
    }
    payload["datasets"] = prepare_data(run_dir)
    payload["status"] = "data_ready"
    _write_summary(run_dir, payload)
    _publish(run_dir, f"Record Part 1 pivot frozen manifests {run_id} [skip ci]")

    payload["loop_position_micro_test"] = run_loop_position_micro_test(run_dir, natural)
    payload["status"] = "micro_test_complete"
    _write_summary(run_dir, payload)
    _publish(run_dir, f"Record loop-position transfer measurement {run_id} [skip ci]")

    payload["branching_screens"] = {
        "natural_step2000_N20_verbal": run_branching_screen(
            run_dir,
            label="natural_step2000_N20_verbal",
            checkpoint=natural,
            data_name="branching_n20_verbal",
        ),
        "n24_step6000_N24_symbolic": run_branching_screen(
            run_dir,
            label="n24_step6000_N24_symbolic",
            checkpoint=n24,
            data_name="branching_n24_symbolic",
        ),
    }
    green = [name for name, result in payload["branching_screens"].items() if result["gate"]["passed"]]
    payload["green_branching_keepers"] = green
    if green:
        payload["phase_g_alpha_status"] = "ready_for_powered_margin_lock_then_launch"
        payload["adapter_decision"] = "not_needed"
    else:
        payload["phase_g_alpha_status"] = "closed_pending_strategy_review"
        payload["adapter_decision"] = (
            "one_detachable_rank_le16_attention_adapter_touchup_requires_explicit_near_miss_review; "
            "no_numeric_near_miss_band_was_preregistered"
        )
    payload["status"] = "finished"
    _write_summary(run_dir, payload)
    _publish(run_dir, f"Finish Part 1 closeout pivot session {run_id} [skip ci]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
