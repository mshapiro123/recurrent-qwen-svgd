"""Run matched forward-table and inverse-table inverse-composition staircases."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
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
from colab.stage5_chain_consolidation_utils import backup_checkpoint_to_drive, path_for_cli  # noqa: E402
from colab.stage5_publish_utils import publishable_artifact_paths  # noqa: E402
from colab.stage5_n24_rung import tier1_canary_verdict  # noqa: E402
from training.abductive_injective_task import (  # noqa: E402
    AbductiveInjectiveConfig,
    build_rows,
    row_manifest,
    with_inverse_table_prompt,
    write_jsonl,
)
from training.staircase_curriculum import (  # noqa: E402
    equalized_loop_weights,
    exposure_fractions,
    optimizer_steps_for_weighted_budget,
)


EXPECTED_TRAIN_SHA = "4ab6377a15d64cf5e07c8855ed05f432feed75e512e196cbd53f648dc9fcb4a5"
EXPECTED_TEST_SHA = "4dd29d9fb7b4170390234646c7c1773377eea56145f6ae659e38f3ae443f2068"
KEEPER_SHA256 = "0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f"
STAGE_BAR_CORRECT = 46
STAGE_BAR_TOTAL = 64
STAGE_BAR = STAGE_BAR_CORRECT / STAGE_BAR_TOTAL
WEIGHTED_LABEL_BUDGET = 1500.0
EVAL_EVERY = 250
PHASE1_CAPS = (2, 3, 4)
PHASE2_CAPS = (5, 6, 7, 8)
PHASE1_STEP_ENVELOPE = 4000
PHASE2_STEP_ENVELOPE = 6000
SYNTHETIC_GUARDRAIL_FLOOR = 0.93
MATCHED_RENDERING_KEYS = {
    "question",
    "prompt",
    "table_direction",
    "display_mapping",
    "control_role",
}
SYNTHETIC_GUARDRAIL_SOURCE = (
    ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
)
NATURAL_CANARY_RELAY = (
    ROOT / "outputs/stage5/stage5_natural_surface_receipts_20260709_210151/data/paired_relay_d1_12.jsonl"
)
NATURAL_CANARY_POINTER = (
    ROOT / "outputs/stage5/stage5_natural_surface_receipts_20260709_210151/data/paired_pointer_d1_12.jsonl"
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _non_rendering_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in MATCHED_RENDERING_KEYS}


def _matched_identity(canonical: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = [
        str(left["id"])
        for left, right in zip(canonical, control)
        if _non_rendering_payload(left) != _non_rendering_payload(right)
    ]
    if len(canonical) != len(control):
        mismatches.append(f"row_count:{len(canonical)}!={len(control)}")
    return {
        "status": "passed" if not mismatches else "failed",
        "rows": len(canonical),
        "allowed_rendering_keys": sorted(MATCHED_RENDERING_KEYS),
        "non_rendering_mismatches": mismatches,
    }


def build_matched_data(data_dir: Path) -> dict[str, Any]:
    train_rows = build_rows(
        AbductiveInjectiveConfig(n_symbols=20, max_depth=8, rows_per_depth=256, seed=1_104_729),
        split="train",
        mode="injective",
    )
    test_rows = build_rows(
        AbductiveInjectiveConfig(n_symbols=20, max_depth=8, rows_per_depth=128, seed=1_104_729),
        split="test",
        mode="injective",
    )
    canonical = {"train": row_manifest(train_rows), "test": row_manifest(test_rows)}
    if canonical["train"]["row_sha256"] != EXPECTED_TRAIN_SHA:
        raise RuntimeError("Regenerated staircase train rows do not match the locked payload hash")
    if canonical["test"]["row_sha256"] != EXPECTED_TEST_SHA:
        raise RuntimeError("Regenerated staircase test rows do not match the locked payload hash")
    control_train = [with_inverse_table_prompt(row) for row in train_rows]
    control_test = [with_inverse_table_prompt(row) for row in test_rows]
    identity_train = _matched_identity(train_rows, control_train)
    identity_test = _matched_identity(test_rows, control_test)
    mismatches = identity_train["non_rendering_mismatches"] + identity_test["non_rendering_mismatches"]
    if mismatches:
        raise RuntimeError(f"Matched arm construction changed non-rendering fields: {mismatches[:5]}")
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("train_F_forward_table", train_rows),
        ("test_F_forward_table", test_rows),
        ("train_C_inverse_table", control_train),
        ("test_C_inverse_table", control_test),
    ):
        write_jsonl(data_dir / f"{name}.jsonl", rows)
    return {
        "canonical": canonical,
        "control": {"train": row_manifest(control_train), "test": row_manifest(control_test)},
        "matched_identity": {
            "status": "passed",
            "train": identity_train,
            "test": identity_test,
            "non_rendering_mismatches": mismatches,
        },
    }


def _rows_through_cap(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["depth"]) <= int(cap)]


def _balanced_prefix(rows: list[dict[str, Any]], *, rows_per_depth: int, max_depth: int) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for row in rows:
        depth = int(row["depth"])
        if depth > int(max_depth) or counts.get(depth, 0) >= int(rows_per_depth):
            continue
        selected.append(row)
        counts[depth] = counts.get(depth, 0) + 1
    expected = {depth: int(rows_per_depth) for depth in range(1, int(max_depth) + 1)}
    if counts != expected:
        raise RuntimeError(f"Could not build balanced prefix: {counts} != {expected}")
    return selected


def stage_training_plan(
    rows: list[dict[str, Any]],
    *,
    cap: int,
    effective_batch_size: int,
    weighted_label_budget: float,
    eval_every: int,
    remaining_phase_steps: int,
) -> dict[str, Any]:
    exposure = exposure_fractions(rows, cap=cap)
    weights = equalized_loop_weights(exposure, cap=cap, newest_multiplier=2.0)
    dose_steps = optimizer_steps_for_weighted_budget(
        weighted_label_budget=weighted_label_budget,
        newest_exposure=exposure[-1],
        newest_weight=weights[-1],
        effective_batch_size=effective_batch_size,
        eval_every=eval_every,
    )
    optimizer_steps = min(int(dose_steps), int(remaining_phase_steps))
    expected_mass = optimizer_steps * effective_batch_size * exposure[-1] * weights[-1]
    return {
        "cap": int(cap),
        "exposure_fractions": exposure,
        "loop_label_weights": weights,
        "newest_multiplier": 2.0,
        "effective_batch_size": int(effective_batch_size),
        "weighted_label_budget": float(weighted_label_budget),
        "eval_every": int(eval_every),
        "dose_limited_optimizer_steps": int(dose_steps),
        "optimizer_steps": int(optimizer_steps),
        "expected_newest_weighted_labels": float(expected_mass),
        "phase_envelope_limited": optimizer_steps < dose_steps,
    }


def assess_stage_gate(summary: dict[str, Any], *, cap: int) -> dict[str, Any]:
    row = summary["test"]["diagonal_by_depth"].get(str(cap), {})
    correct = int(row.get("correct", 0))
    total = int(row.get("total", 0))
    return {
        "cap": int(cap),
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "required_correct": STAGE_BAR_CORRECT,
        "required_total": STAGE_BAR_TOTAL,
        "passed": total == STAGE_BAR_TOTAL and correct >= STAGE_BAR_CORRECT,
    }


def classify_matched_arms(
    *,
    experiment_dose: float | None,
    control_dose: float | None,
) -> str:
    if experiment_dose is not None and control_dose is None:
        return "instrumentation_alarm"
    if experiment_dose is None and control_dose is None:
        return "composition_hard_both"
    if experiment_dose is None:
        return "non_native_position_cost"
    assert control_dose is not None
    if control_dose <= 0.0:
        return "instrumentation_alarm"
    if experiment_dose >= 0.8 * WEIGHTED_LABEL_BUDGET and control_dose >= 0.8 * WEIGHTED_LABEL_BUDGET:
        return "composition_hard_both"
    if experiment_dose / control_dose >= 5.0:
        return "non_native_position_cost"
    return "exposure_starvation"


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Could not parse checkpoint step from {path}")
    return int(match.group(1))


def _dose_at_step(training_summary: dict[str, Any], step: int, cap: int) -> dict[str, Any]:
    matches = [row for row in training_summary.get("dose_trace", []) if int(row.get("step", -1)) == int(step)]
    if not matches:
        raise RuntimeError(f"Training summary has no dose receipt at checkpoint step {step}")
    row = matches[-1]
    return {
        "step": int(step),
        "raw_active_labels": list(row["raw_active_labels"]),
        "weighted_active_labels": list(row["weighted_active_labels"]),
        "newest_raw_active_labels": int(row["raw_active_labels"][cap - 1]),
        "newest_weighted_active_labels": float(row["weighted_active_labels"][cap - 1]),
        "equalization_ratios": list(row["equalization_ratios"]),
    }


def _write_stage_config(
    path: Path,
    *,
    checkpoint: Path,
    output_dir: Path,
    plan: dict[str, Any],
    seed: int,
) -> None:
    cap = int(plan["cap"])
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("STAGE5_STAIRCASE_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 640,
        "max_loops": cap,
        "loop_loss_mode": "weighted_per_loop_labels",
        "loop_label_weights": plan["loop_label_weights"],
        "newest_loop_multiplier": 2.0,
        "dose_assert_every": 200,
        "dose_ratio_min": 0.8,
        "dose_ratio_max": 1.25,
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "minimum_effective_batch_size": 8,
        "seed": int(seed),
        "optimizer": "adamw",
        "learning_rate": float(os.environ.get("STAGE5_STAIRCASE_LR", "1e-5")),
        "adamw_lr": float(os.environ.get("STAGE5_STAIRCASE_LR", "1e-5")),
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": int(plan["optimizer_steps"]),
        "save_every": EVAL_EVERY,
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
        "require_lora_loaded_before_merge": False,
        "train_auxiliary": {
            "bridge": True,
            "halting": False,
            "reentry_adapter": False,
            "latent": False,
        },
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": cap,
            "end_loop": cap,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "synthetic_phase": "inverse_composition_staircase",
        "synthetic_stage": f"cap_{cap}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _run_staircase_eval(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    train_jsonl: Path,
    test_jsonl: Path,
    cap: int,
    probes: bool,
    cka: bool = False,
    include_train: bool = False,
) -> dict[str, Any]:
    output_dir = run_dir / "eval" / label
    command = [
        sys.executable,
        "eval/eval_abductive_staircase.py",
        "--train_jsonl",
        path_for_cli(train_jsonl),
        "--test_jsonl",
        path_for_cli(test_jsonl),
        "--checkpoint",
        path_for_cli(checkpoint),
        "--output_dir",
        path_for_cli(output_dir),
        "--rows_per_depth",
        "64",
        "--max_loops",
        str(cap),
        "--permutations",
        os.environ.get("STAGE5_STAIRCASE_PROBE_PERMUTATIONS", "100"),
        "--bridge_projection_mode",
        "split",
        "--dtype",
        os.environ.get("STAGE5_STAIRCASE_DTYPE", "bfloat16"),
        "--adapter_dtype",
        "float32",
        "--device",
        os.environ.get("DEVICE", "cuda"),
    ]
    command.append("--run_probes" if probes else "--no-run_probes")
    command.append("--include_train_predictions" if include_train else "--no-include_train_predictions")
    if cka:
        command.append("--run_cka")
    run(command)
    return read_json(output_dir / "summary.json")


def _run_diagonal(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    data_jsonl: Path,
    max_depth: int,
    value_prefix: str,
) -> dict[str, Any]:
    output_dir = run_dir / "guardrails" / label
    run(
        [
            sys.executable,
            "eval/eval_synthetic_diagonal_guardrail.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(output_dir / "rows.jsonl"),
            "--output_summary",
            path_for_cli(output_dir / "summary.json"),
            "--max_depth",
            str(max_depth),
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_STAIRCASE_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    return read_json(output_dir / "summary.json")


def _guardrail_receipt(run_dir: Path, *, label: str, checkpoint: Path, data_jsonl: Path) -> dict[str, Any]:
    result = _run_diagonal(
        run_dir,
        label=label,
        checkpoint=checkpoint,
        data_jsonl=data_jsonl,
        max_depth=12,
        value_prefix="letter:",
    )
    return {
        "summary": path_for_cli(run_dir / "guardrails" / label / "summary.json"),
        "active_diagonal_min": float(result["active_diagonal_min"]),
        "floor": SYNTHETIC_GUARDRAIL_FLOOR,
        "passed": float(result["active_diagonal_min"]) >= SYNTHETIC_GUARDRAIL_FLOOR,
    }


def _publish(run_dir: Path, message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        relative_parts = path.relative_to(run_dir).parts
        if any(part in {"data", "stage_data", "guardrail_data"} for part in relative_parts):
            continue
        if path.name == "rows.jsonl":
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def _write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Inverse Composition Staircase - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Keeper SHA256: `{payload.get('keeper', {}).get('checkpoint_sha256')}`",
        f"- Phase 2 opened: `{payload.get('phase2_opened', False)}`",
        f"- Matched-arm reading: `{payload.get('matched_arm_reading')}`",
        "",
        "## Stage table",
        "",
        "| Arm | Cap | Passed | Correct | Weighted labels to bar | Guardrail |",
        "|---|---:|---|---:|---:|---|",
    ]
    for arm_name, arm in payload.get("arms", {}).items():
        for stage in arm.get("stages", []):
            gate = stage.get("gate", {})
            dose = stage.get("dose_to_bar") or {}
            guard = stage.get("synthetic_guardrail", {})
            lines.append(
                f"| {arm_name} | {stage['cap']} | {gate.get('passed')} | "
                f"{gate.get('correct', 0)}/{gate.get('total', 0)} | "
                f"{dose.get('newest_weighted_active_labels')} | {guard.get('passed')} |"
            )
    lines.extend(
        [
            "",
            "The active matrix is primary. Final diagonal is secondary. Phase G-alpha remains closed.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def _prepare_guardrail_data(run_dir: Path) -> dict[str, Path]:
    data_dir = run_dir / "guardrail_data"
    synthetic = _balanced_prefix(read_jsonl(SYNTHETIC_GUARDRAIL_SOURCE), rows_per_depth=32, max_depth=12)
    natural = _balanced_prefix(read_jsonl(NATURAL_CANARY_RELAY), rows_per_depth=16, max_depth=8)
    natural.extend(_balanced_prefix(read_jsonl(NATURAL_CANARY_POINTER), rows_per_depth=16, max_depth=8))
    synthetic_path = data_dir / "synthetic_frozen_32_per_depth_d1_12.jsonl"
    natural_path = data_dir / "natural_surface_relay_pointer_16_each_d1_8.jsonl"
    write_jsonl(synthetic_path, synthetic)
    write_jsonl(natural_path, natural)
    return {"synthetic": synthetic_path, "natural": natural_path}


def _stage_checkpoint_candidates(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("unfrozen_recurrent_step_*.pt"), key=_checkpoint_step)


def _run_stage(
    run_dir: Path,
    *,
    arm_name: str,
    cap: int,
    checkpoint: Path,
    train_rows: list[dict[str, Any]],
    test_jsonl: Path,
    phase_remaining: int,
    seed: int,
) -> dict[str, Any]:
    stage_rows = _rows_through_cap(train_rows, cap)
    stage_dir = run_dir / "train" / arm_name / f"cap_{cap}"
    train_jsonl = run_dir / "stage_data" / arm_name / f"train_cap_{cap}.jsonl"
    write_jsonl(train_jsonl, stage_rows)
    plan = stage_training_plan(
        stage_rows,
        cap=cap,
        effective_batch_size=8,
        weighted_label_budget=WEIGHTED_LABEL_BUDGET,
        eval_every=EVAL_EVERY,
        remaining_phase_steps=phase_remaining,
    )
    if plan["optimizer_steps"] <= 0:
        return {"cap": cap, "status": "phase_step_envelope_exhausted", "plan": plan, "gate": {"passed": False}}
    config = run_dir / "configs" / arm_name / f"cap_{cap}.yaml"
    _write_stage_config(config, checkpoint=checkpoint, output_dir=stage_dir, plan=plan, seed=seed)
    process = run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config),
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    log_path = stage_dir / "train.log"
    log_path.write_text(process.stdout or "", encoding="utf-8")
    train_summary = read_json(stage_dir / "train_unfrozen_recurrent_summary.json")
    checkpoints = _stage_checkpoint_candidates(stage_dir)
    if not checkpoints:
        raise RuntimeError(f"Stage produced no checkpoints: {stage_dir}")

    checkpoint_evals: list[dict[str, Any]] = []
    selected: Path | None = None
    selected_gate: dict[str, Any] | None = None
    for candidate in checkpoints:
        step = _checkpoint_step(candidate)
        eval_summary = _run_staircase_eval(
            run_dir,
            label=f"{arm_name}_cap{cap}_step{step}_gate",
            checkpoint=candidate,
            train_jsonl=train_jsonl,
            test_jsonl=test_jsonl,
            cap=cap,
            probes=False,
        )
        gate = assess_stage_gate(eval_summary, cap=cap)
        dose = _dose_at_step(train_summary, step, cap)
        checkpoint_evals.append(
            {
                "step": step,
                "checkpoint": path_for_cli(candidate),
                "gate": gate,
                "dose": dose,
                "summary": path_for_cli(run_dir / "eval" / f"{arm_name}_cap{cap}_step{step}_gate" / "summary.json"),
            }
        )
        if gate["passed"] and dose["newest_weighted_active_labels"] <= plan["expected_newest_weighted_labels"] + 1e-6:
            selected = candidate
            selected_gate = gate
            break
    if selected is None:
        selected = checkpoints[-1]
        selected_gate = checkpoint_evals[-1]["gate"]
    selected_step = _checkpoint_step(selected)
    selected_dose = _dose_at_step(train_summary, selected_step, cap)
    diagnostics = _run_staircase_eval(
        run_dir,
        label=f"{arm_name}_cap{cap}_selected_diagnostic",
        checkpoint=selected,
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        cap=cap,
        probes=True,
        cka=cap == 4,
        include_train=not bool(selected_gate["passed"]),
    )
    drive_backup = backup_checkpoint_to_drive(
        selected,
        run_id=run_dir.name,
        stage_name=f"{arm_name}_cap_{cap}_selected",
        enabled=True,
    )
    selected_sha = sha256_file(selected)
    for candidate in checkpoints:
        if candidate != selected:
            candidate.unlink(missing_ok=True)
    return {
        "cap": cap,
        "status": "advanced" if selected_gate["passed"] else "stalled",
        "plan": plan,
        "optimizer_steps_spent": int(plan["optimizer_steps"]),
        "optimizer_state_policy": "preserved_within_stage_restarted_identically_at_stage_boundary",
        "checkpoint_evals": checkpoint_evals,
        "selected_step": selected_step,
        "checkpoint": path_for_cli(selected),
        "checkpoint_sha256": selected_sha,
        "checkpoint_drive_backup": drive_backup,
        "gate": selected_gate,
        "dose_to_bar": selected_dose if selected_gate["passed"] else None,
        "final_dose": _dose_at_step(train_summary, _checkpoint_step(checkpoints[-1]), cap),
        "diagnostic_summary": path_for_cli(run_dir / "eval" / f"{arm_name}_cap{cap}_selected_diagnostic" / "summary.json"),
        "conditional_transition_success": diagnostics["conditional_transition_success"],
        "target_decodability": diagnostics["target_decodability"],
        "stratified_loop2_cka": diagnostics.get("stratified_loop2_cka"),
    }


def _restore_stage_checkpoint(run_dir: Path, arm_name: str, stage: dict[str, Any]) -> Path:
    checkpoint, receipt = restore_checkpoint(
        [stage.get("checkpoint_drive_backup"), stage.get("checkpoint")],
        run_dir / "restored" / f"{arm_name}_cap{stage['cap']}.pt",
        label=f"staircase_{arm_name}_cap{stage['cap']}",
    )
    if receipt["selected_checkpoint_sha256"] != stage["checkpoint_sha256"]:
        raise RuntimeError(f"Restored stage checkpoint SHA mismatch for {arm_name} cap {stage['cap']}")
    return checkpoint


def latest_checkpoint_stage(stages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the newest stage that actually produced a durable checkpoint."""

    return next(
        (
            stage
            for stage in reversed(stages)
            if stage.get("checkpoint_drive_backup") and stage.get("checkpoint_sha256")
        ),
        None,
    )


def _paired_reading(payload: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    arm_f = {int(stage["cap"]): stage for stage in payload["arms"]["F"]["stages"]}
    arm_c = {int(stage["cap"]): stage for stage in payload["arms"]["C"]["stages"]}
    for cap in sorted(set(arm_f).intersection(arm_c)):
        f_dose = (arm_f[cap].get("dose_to_bar") or {}).get("newest_weighted_active_labels")
        c_dose = (arm_c[cap].get("dose_to_bar") or {}).get("newest_weighted_active_labels")
        rows[str(cap)] = {
            "experiment_weighted_labels_to_bar": f_dose,
            "control_weighted_labels_to_bar": c_dose,
            "ratio": (float(f_dose) / float(c_dose)) if f_dose is not None and c_dose else None,
            "reading": classify_matched_arms(experiment_dose=f_dose, control_dose=c_dose),
        }
    return rows


def main() -> int:
    run_id = os.environ.get("STAGE5_STAIRCASE_RUN_ID") or time.strftime(
        "stage5_inverse_composition_staircase_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    payload = read_json(summary_path) if summary_path.exists() else {
        "kind": "stage5_inverse_composition_staircase",
        "run_id": run_id,
        "status": "started",
        "phase_g_alpha_status": "closed",
        "arms": {
            "F": {"role": "forward_table_inverse_composition_experiment", "stages": []},
            "C": {"role": "inverse_table_forward_lookup_control", "stages": []},
        },
    }

    receipt_path = Path(os.environ.get("STAGE5_STAIRCASE_KEEPER_RECEIPT", str(DEFAULT_RECEIPT)))
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    keeper_gate = keeper_gate_from_receipt(read_json(receipt_path))
    if keeper_gate["status"] != "green" or keeper_gate["checkpoint_sha256"] != KEEPER_SHA256:
        raise RuntimeError(f"Locked staircase keeper is not the preregistered green checkpoint: {keeper_gate}")
    keeper, keeper_restore = restore_checkpoint(
        [keeper_gate["checkpoint"]],
        run_dir / "restored" / "locked_natural_surface_step2000.pt",
        label="staircase_locked_keeper",
    )
    if keeper_restore["selected_checkpoint_sha256"] != KEEPER_SHA256:
        raise RuntimeError("Restored staircase keeper SHA mismatch")
    print(f"[assert-ok] staircase_keeper_sha256={KEEPER_SHA256}", flush=True)
    payload["keeper"] = keeper_gate

    if "datasets" not in payload:
        payload["datasets"] = build_matched_data(run_dir / "data")
        guardrail_paths = _prepare_guardrail_data(run_dir)
        payload["guardrail_data"] = {key: path_for_cli(value) for key, value in guardrail_paths.items()}
        payload["status"] = "data_ready"
        _write_summary(run_dir, payload)
        _publish(run_dir, f"Record inverse-composition staircase data {run_id} [skip ci]")
    guardrail_paths = {key: ROOT / value for key, value in payload["guardrail_data"].items()}

    train_rows = {
        "F": read_jsonl(run_dir / "data" / "train_F_forward_table.jsonl"),
        "C": read_jsonl(run_dir / "data" / "train_C_inverse_table.jsonl"),
    }
    test_paths = {
        "F": run_dir / "data" / "test_F_forward_table.jsonl",
        "C": run_dir / "data" / "test_C_inverse_table.jsonl",
    }

    # The baseline natural-surface canary is evaluated once on the exact rows
    # used for every later 1,000-step comparison.
    if "tier1_canary_baseline" not in payload:
        baseline = _run_diagonal(
            run_dir,
            label="tier1_natural_surface_keeper",
            checkpoint=keeper,
            data_jsonl=guardrail_paths["natural"],
            max_depth=8,
            value_prefix="name:",
        )
        payload["tier1_canary_baseline"] = {
            "accuracy": float(baseline["accuracy"]),
            "active_diagonal_min": float(baseline["active_diagonal_min"]),
            "summary": path_for_cli(run_dir / "guardrails" / "tier1_natural_surface_keeper" / "summary.json"),
        }
        _write_summary(run_dir, payload)

    current_checkpoints = {"F": keeper, "C": keeper}
    for arm_name in ("F", "C"):
        checkpoint_stage = latest_checkpoint_stage(payload["arms"][arm_name]["stages"])
        if checkpoint_stage is not None:
            current_checkpoints[arm_name] = _restore_stage_checkpoint(
                run_dir,
                arm_name,
                checkpoint_stage,
            )

    for phase_name, caps, envelope in (
        ("phase1", PHASE1_CAPS, PHASE1_STEP_ENVELOPE),
        ("phase2", PHASE2_CAPS, PHASE2_STEP_ENVELOPE),
    ):
        if phase_name == "phase2":
            phase1_green = all(
                [stage["gate"]["passed"] for arm in payload["arms"].values() for stage in arm["stages"] if stage["cap"] in PHASE1_CAPS]
            ) and all(
                len([stage for stage in arm["stages"] if stage["cap"] in PHASE1_CAPS]) == len(PHASE1_CAPS)
                for arm in payload["arms"].values()
            )
            continuation_green = all(
                bool(payload["arms"][arm]["stages"][-1].get("synthetic_guardrail", {}).get("passed"))
                for arm in ("F", "C")
            )
            payload["phase2_opened"] = bool(phase1_green and continuation_green)
            if not payload["phase2_opened"]:
                break
        for cap in caps:
            for arm_index, arm_name in enumerate(("F", "C")):
                existing = next(
                    (stage for stage in payload["arms"][arm_name]["stages"] if int(stage["cap"]) == cap),
                    None,
                )
                if existing is not None:
                    if latest_checkpoint_stage([existing]) is not None:
                        current_checkpoints[arm_name] = _restore_stage_checkpoint(run_dir, arm_name, existing)
                    continue
                spent = sum(
                    int(stage.get("optimizer_steps_spent", 0))
                    for stage in payload["arms"][arm_name]["stages"]
                    if int(stage["cap"]) in caps
                )
                stage = _run_stage(
                    run_dir,
                    arm_name=arm_name,
                    cap=cap,
                    checkpoint=current_checkpoints[arm_name],
                    train_rows=train_rows[arm_name],
                    test_jsonl=test_paths[arm_name],
                    phase_remaining=envelope - spent,
                    seed=81_001 + cap * 100 + arm_index,
                )
                stage_has_checkpoint = bool(stage.get("checkpoint_drive_backup"))
                if stage_has_checkpoint:
                    current_checkpoints[arm_name] = _restore_stage_checkpoint(run_dir, arm_name, stage)
                    stage["synthetic_guardrail"] = _guardrail_receipt(
                        run_dir,
                        label=f"{arm_name}_cap{cap}_synthetic",
                        checkpoint=current_checkpoints[arm_name],
                        data_jsonl=guardrail_paths["synthetic"],
                    )
                else:
                    # A phase-envelope stop has no newly trained checkpoint. Do
                    # not score the preceding stage's checkpoint as this stage.
                    stage["synthetic_guardrail"] = {
                        "status": "not_run_no_stage_checkpoint",
                        "passed": False,
                    }
                if not stage["synthetic_guardrail"]["passed"]:
                    if stage_has_checkpoint:
                        stage["status"] = "blocked_synthetic_guardrail"
                    stage["gate"]["passed"] = False
                previous_spent = sum(
                    int(item.get("optimizer_steps_spent", 0))
                    for item in payload["arms"][arm_name]["stages"]
                )
                cumulative_spent = previous_spent + int(stage.get("optimizer_steps_spent", 0))
                prior_canaries = list(payload["arms"][arm_name].get("tier1_canaries") or [])
                milestones_due = cumulative_spent // 1000 if stage_has_checkpoint else len(prior_canaries)
                while len(prior_canaries) < milestones_due:
                    milestone = (len(prior_canaries) + 1) * 1000
                    canary = _run_diagonal(
                        run_dir,
                        label=f"{arm_name}_tier1_milestone_{milestone}",
                        checkpoint=current_checkpoints[arm_name],
                        data_jsonl=guardrail_paths["natural"],
                        max_depth=8,
                        value_prefix="name:",
                    )
                    baseline_accuracy = float(payload["tier1_canary_baseline"]["accuracy"])
                    accuracy_delta = float(canary["accuracy"]) - baseline_accuracy
                    verdict = tier1_canary_verdict(
                        accuracy_delta=accuracy_delta,
                        ppl_relative_delta=None,
                    )
                    prior_canaries.append(
                        {
                            "milestone_optimizer_steps": milestone,
                            "observed_at_stage_cap": cap,
                            "candidate_accuracy": float(canary["accuracy"]),
                            "baseline_accuracy": baseline_accuracy,
                            "accuracy_delta": accuracy_delta,
                            "verdict": verdict,
                            "summary": path_for_cli(
                                run_dir / "guardrails" / f"{arm_name}_tier1_milestone_{milestone}" / "summary.json"
                            ),
                        }
                    )
                    if verdict["status"] == "red_hard_stop":
                        stage["status"] = "blocked_tier1_canary"
                        stage["gate"]["passed"] = False
                        break
                payload["arms"][arm_name]["tier1_canaries"] = prior_canaries
                payload["arms"][arm_name]["stages"].append(stage)
                payload["status"] = f"{phase_name}_{arm_name}_cap{cap}_{stage['status']}"
                payload["matched_arm_reading"] = _paired_reading(payload)
                _write_summary(run_dir, payload)
                _publish(
                    run_dir,
                    f"Record inverse-composition staircase {arm_name} cap {cap} {run_id} [skip ci]",
                )
            cap_rows = [
                next(stage for stage in payload["arms"][arm]["stages"] if int(stage["cap"]) == cap)
                for arm in ("F", "C")
            ]
            if not all(bool(stage["gate"]["passed"]) for stage in cap_rows):
                payload["status"] = f"blocked_at_cap_{cap}"
                break
        if payload["status"].startswith("blocked_at_cap_"):
            break

    payload["matched_arm_reading"] = _paired_reading(payload)
    payload["phase_g_alpha_status"] = "closed_pending_deterministic_gate"
    payload["status"] = (
        "staircase_completed_depth8"
        if all(
            len(arm["stages"]) == len(PHASE1_CAPS) + len(PHASE2_CAPS)
            and all(stage["gate"]["passed"] for stage in arm["stages"])
            for arm in payload["arms"].values()
        )
        else payload["status"]
    )
    _write_summary(run_dir, payload)
    _publish(run_dir, f"Finish inverse-composition staircase {run_id} [skip ci]")
    return 0 if payload["status"] == "staircase_completed_depth8" else 2


if __name__ == "__main__":
    raise SystemExit(main())
