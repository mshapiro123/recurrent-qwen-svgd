"""One bounded deterministic continuation for the inverse-rendered N24 gate.

This is W4 from the two-lane rebase.  It deliberately trains no stochastic
heads, leaves the frozen test split sealed until calibration plus guardrails
pass, and uses no forward rehearsal after the cap-3 replay showed that the
fixed 25% rehearsal mix was not competence preserving.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_inverse_composition_staircase import (  # noqa: E402
    _guardrail_receipt,
    _prepare_guardrail_data,
    _publish,
    _run_diagonal,
)
from colab.run_stage5_inverse_rendered_width_gate import (  # noqa: E402
    DATA_ROOT,
    SOURCE_SUMMARY,
    assess_deterministic_validity,
    resolve_keeper_source,
)
from colab.run_stage5_natural_surface_transfer import restore_checkpoint, run, sha256_file  # noqa: E402
from colab.stage5_n24_rung import tier1_canary_verdict  # noqa: E402
from colab.stage5_chain_consolidation_utils import backup_checkpoint_to_drive, path_for_cli  # noqa: E402
from training.abductive_injective_task import (  # noqa: E402
    PhaseGFrozenEvalConfig,
    build_phase_g_frozen_rows,
    row_manifest,
    validate_inverse_relation_rows,
    with_inverse_relation_prompt,
    write_jsonl,
)
from training.staircase_curriculum import equalized_loop_weights, exposure_fractions  # noqa: E402


RUN_PREFIX = "stage5_inverse_rendered_n24_continuation"
TRAIN_ROWS_PER_STRATUM = 256
TRAIN_SEED = 9_217_347
MAX_LOOPS = 4
EFFECTIVE_BATCH_SIZE = 8
MAX_STEPS = 200
LEARNING_RATE = 1e-5
NATURAL_BASELINE_SUMMARY = ROOT / "outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_causal_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Continuation training rows are empty")
    for row in rows:
        if not str(row.get("prompt") or "").strip():
            raise RuntimeError(f"{row.get('id')}: missing prompt")
        if not str(row.get("completion") or "").strip():
            raise RuntimeError(f"{row.get('id')}: missing completion")
        if len(row.get("loop_completions") or []) != int(row.get("depth", 0)):
            raise RuntimeError(f"{row.get('id')}: loop completion/depth mismatch")


def prepare_training_rows(run_dir: Path, *, rows_per_stratum: int = TRAIN_ROWS_PER_STRATUM) -> dict[str, Any]:
    """Build independent arbitrary N24 training rows and preserve frozen eval IDs."""

    config = PhaseGFrozenEvalConfig(rows_per_stratum=int(rows_per_stratum), seed=TRAIN_SEED)
    canonical_rows = build_phase_g_frozen_rows(config, split="train")
    rows = [with_inverse_relation_prompt(row) for row in canonical_rows]
    validation = validate_inverse_relation_rows(rows, rows_per_stratum=int(rows_per_stratum))
    if validation["status"] != "passed":
        raise RuntimeError(f"N24 continuation training data failed validation: {validation['errors'][:5]}")
    validate_causal_rows(rows)

    frozen_ids: set[str] = set()
    for split in ("calibration", "test"):
        frozen_path = DATA_ROOT / f"{split}_n24.jsonl"
        if not frozen_path.exists():
            raise FileNotFoundError(f"Missing frozen inverse-rendered {split} set: {frozen_path}")
        frozen_ids.update(str(row["id"]) for row in read_jsonl(frozen_path))
    overlap = sorted(frozen_ids.intersection(str(row["id"]) for row in rows))
    if overlap:
        raise RuntimeError(f"Continuation training overlaps frozen evaluation IDs: {overlap[:3]}")

    data_dir = run_dir / "data"
    train_path = data_dir / "train_n24_inverse_relation.jsonl"
    write_jsonl(train_path, rows)
    receipt = {
        "path": path_for_cli(train_path),
        "manifest": row_manifest(rows),
        "validation": validation,
        "rows_per_stratum": int(rows_per_stratum),
        "seed": TRAIN_SEED,
        "rendering": "inverse_relation_given",
        "forward_rehearsal_fraction": 0.0,
        "forward_rehearsal_decision": "disabled_after_cap3_replay_no_joint_gate_candidate",
        "frozen_eval_id_overlap": len(overlap),
    }
    write_json(data_dir / "training_manifest.json", receipt)
    return receipt


def loop_weights(rows: list[dict[str, Any]]) -> list[float]:
    exposure = exposure_fractions([{"depth": int(row["depth"])} for row in rows], cap=MAX_LOOPS)
    return equalized_loop_weights(exposure, cap=MAX_LOOPS, newest_multiplier=2.0)


def write_config(
    path: Path,
    *,
    checkpoint: Path,
    output_dir: Path,
    checkpoint_backup_dir: Path,
    progress_backup_path: Path,
) -> None:
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": os.environ.get("STAGE5_INVERSE_RENDERED_CONTINUATION_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 768,
        "max_loops": MAX_LOOPS,
        "loop_loss_mode": "weighted_per_loop_labels",
        "row_specific_forward_loops": True,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE,
        "minimum_effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "seed": TRAIN_SEED,
        "optimizer": "adamw",
        "learning_rate": LEARNING_RATE,
        "adamw_lr": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": MAX_STEPS,
        "save_every": 50,
        "save_steps": [100, 150],
        "checkpoint_backup_every": 50,
        "checkpoint_backup_dir": str(checkpoint_backup_dir),
        "progress_backup_path": str(progress_backup_path),
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
        "train_auxiliary": {"bridge": True, "halting": False, "reentry_adapter": False, "latent": False},
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": MAX_LOOPS,
            "end_loop": MAX_LOOPS,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "synthetic_phase": "inverse_rendered_n24_continuation",
        "synthetic_stage": "w4_bounded_deterministic_tune",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_eval(run_dir: Path, *, label: str, checkpoint: Path, data_jsonl: Path) -> dict[str, Any]:
    eval_dir = run_dir / "eval" / label
    run(
        [
            sys.executable,
            "eval/eval_abductive_coverage.py",
            "--data_jsonl", path_for_cli(data_jsonl),
            "--checkpoint", path_for_cli(checkpoint),
            "--output_jsonl", path_for_cli(eval_dir / "rows.jsonl"),
            "--output_summary", path_for_cli(eval_dir / "summary.json"),
            "--sample_counts", "1",
            "--temperature", "0.7",
            "--bridge_projection_mode", "split",
            "--dtype", os.environ.get("STAGE5_INVERSE_RENDERED_CONTINUATION_DTYPE", "bfloat16"),
            "--adapter_dtype", "float32",
            "--device", os.environ.get("DEVICE", "cuda"),
            "--progress_every", "16",
        ],
        cwd=ROOT,
    )
    return read_json(eval_dir / "summary.json")


def main() -> int:
    run_id = os.environ.get("STAGE5_INVERSE_RENDERED_CONTINUATION_RUN_ID") or time.strftime(f"{RUN_PREFIX}_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(os.environ.get("STAGE5_INVERSE_RENDERED_CONTINUATION_SOURCE", str(SOURCE_SUMMARY)))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source = read_json(source_path)
    keeper = resolve_keeper_source(source)
    checkpoint, restore = restore_checkpoint(
        keeper["checkpoint_candidates"],
        run_dir / "restored" / "C_cap3.pt",
        label="inverse_rendered_n24_continuation_source",
    )
    if restore["selected_checkpoint_sha256"] != keeper["checkpoint_sha256"]:
        raise RuntimeError("Continuation source checkpoint SHA mismatch")

    train_receipt = prepare_training_rows(run_dir)
    train_rows = read_jsonl(ROOT / train_receipt["path"])
    weights = loop_weights(train_rows)
    for row in train_rows:
        row["loop_label_weights"] = weights
        row["forward_loop_count"] = int(row["depth"])
    train_path = ROOT / train_receipt["path"]
    write_jsonl(train_path, train_rows)
    train_receipt["loop_label_weights"] = weights
    write_json(run_dir / "data" / "training_manifest.json", train_receipt)

    train_dir = run_dir / "train" / "continuation"
    drive_root = Path(os.environ.get("STAGE5_INVERSE_RENDERED_CONTINUATION_DRIVE_ROOT", f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{run_id}/train"))
    config_path = run_dir / "config" / "continuation.yaml"
    write_config(
        config_path,
        checkpoint=checkpoint,
        output_dir=train_dir,
        checkpoint_backup_dir=drive_root / "checkpoints",
        progress_backup_path=drive_root / "train_progress.json",
    )
    run([sys.executable, "training/train_unfrozen_recurrent.py", "--config", path_for_cli(config_path), "--train_jsonl", path_for_cli(train_path), "--device", os.environ.get("DEVICE", "cuda")], cwd=ROOT)
    trained = train_dir / f"unfrozen_recurrent_step_{MAX_STEPS}.pt"
    if not trained.exists():
        raise RuntimeError(f"Training did not produce {trained}")
    checkpoint_sha = sha256_file(trained)
    drive_backup = backup_checkpoint_to_drive(trained, run_id=run_id, stage_name="selected", enabled=True)
    _publish(run_dir, f"Record inverse-rendered N24 continuation train {run_id} [skip ci]")

    calibration = assess_deterministic_validity(run_eval(run_dir, label="calibration", checkpoint=trained, data_jsonl=DATA_ROOT / "calibration_n24.jsonl"))
    guardrail_paths = _prepare_guardrail_data(run_dir)
    synthetic = _guardrail_receipt(run_dir, label="synthetic", checkpoint=trained, data_jsonl=guardrail_paths["synthetic"])
    natural = _run_diagonal(run_dir, label="natural", checkpoint=trained, data_jsonl=guardrail_paths["natural"], max_depth=8, value_prefix="name:")
    baseline = float(read_json(NATURAL_BASELINE_SUMMARY)["tier1_canary_baseline"]["accuracy"])
    delta = float(natural["accuracy"]) - baseline
    natural_gate = {
        "baseline_accuracy": baseline,
        "candidate_accuracy": float(natural["accuracy"]),
        "accuracy_delta": delta,
        "verdict": tier1_canary_verdict(accuracy_delta=delta, ppl_relative_delta=None),
    }
    natural_gate["passed"] = natural_gate["verdict"]["status"] != "red_hard_stop"

    test_summary = None
    test_gate = None
    if calibration["pass"] and synthetic["passed"] and natural_gate["passed"]:
        test_summary = run_eval(run_dir, label="test", checkpoint=trained, data_jsonl=DATA_ROOT / "test_n24.jsonl")
        test_gate = assess_deterministic_validity(test_summary)

    green = bool(calibration["pass"] and synthetic["passed"] and natural_gate["passed"] and test_gate and test_gate["pass"])
    status = "deterministic_gate_green" if green else "bounded_tune_review_required"
    payload = {
        "kind": "stage5_inverse_rendered_n24_continuation",
        "run_id": run_id,
        "status": status,
        "source_summary": path_for_cli(source_path),
        "source_checkpoint_sha256": keeper["checkpoint_sha256"],
        "restore_receipt": restore,
        "checkpoint": path_for_cli(trained),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_drive_backup": drive_backup,
        "training": {"steps": MAX_STEPS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "optimizer": "adamw", **train_receipt},
        "calibration": calibration,
        "synthetic_retention": synthetic,
        "natural_canary": natural_gate,
        "test": {"opened": test_summary is not None, "summary": test_summary, "gate": test_gate},
        "phase_g_alpha_status": "open_for_k1_parity" if green else "closed",
        "cap4_authorized": False,
        "do_not_claim": "This bounded deterministic tune is a G-alpha substrate gate, not evidence that stochastic width helps.",
    }
    write_json(run_dir / "summary.json", payload)
    (run_dir / "summary.md").write_text(
        "\n".join([
            f"# Inverse-Rendered N24 Continuation - {run_id}", "",
            f"- Status: `{status}`",
            f"- Calibration: `{calibration['pooled']['correct']}/{calibration['pooled']['total']}`",
            f"- Synthetic retention: `{synthetic['active_diagonal_min']}`",
            f"- Natural canary: `{natural_gate['verdict']['status']}`",
            f"- Test opened: `{test_summary is not None}`",
            f"- Phase G-alpha: `{payload['phase_g_alpha_status']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    _publish(run_dir, f"Finish inverse-rendered N24 continuation {run_id} [skip ci]")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if green else 2


if __name__ == "__main__":
    raise SystemExit(main())
