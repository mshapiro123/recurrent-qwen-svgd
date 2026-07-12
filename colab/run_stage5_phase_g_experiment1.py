"""Run the deterministic Phase G task gate and answer-head coverage baseline."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import restore_checkpoint, run, sha256_file
from colab.run_stage5_phase_g_gate_prepare import (
    DEFAULT_RECEIPT,
    generate_gate_data,
    keeper_gate_from_receipt,
)
from colab.stage5_chain_consolidation_utils import (
    backup_checkpoint_to_drive,
    latest_checkpoint,
    path_for_cli,
)
from colab.stage5_publish_utils import publishable_artifact_paths
from training.abductive_injective_task import write_jsonl


SYNTHETIC_GUARDRAIL_DATA = (
    ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
)
INJECTIVE_POOLED_FLOOR = 0.90
INJECTIVE_DEPTH_FLOOR = 0.80
ABDUCTIVE_POOLED_FLOOR = 0.75
ABDUCTIVE_DEPTH_FLOOR = 0.60
SYNTHETIC_FLOOR = 0.93


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Phase G Experiment 1 - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Keeper: `{payload['keeper_gate']['checkpoint']}`",
        f"- Keeper SHA256: `{payload['keeper_gate']['checkpoint_sha256']}`",
        f"- Injective gate: `{payload.get('injective_gate')}`",
        f"- Abductive gate: `{payload.get('abductive_gate')}`",
        f"- Synthetic guardrail: `{payload.get('synthetic_guardrail')}`",
        "",
        "LPRM, learned halting, latent width, and SVGD remain disabled in this experiment.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def publish_lightweight(run_dir: Path, *, message: str) -> None:
    paths = []
    for path in publishable_artifact_paths(run_dir):
        rel = path.relative_to(run_dir)
        if rel.parts and rel.parts[0] == "data":
            continue
        if path.name.endswith("_rows.jsonl"):
            continue
        paths.append(path)
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in paths:
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print(f"No new lightweight artifacts for {run_dir}", flush=True)
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def write_arm_config(
    run_dir: Path,
    *,
    arm: str,
    keeper: Path,
    max_steps: int,
    seed: int,
) -> Path:
    output_dir = run_dir / "train" / arm
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("STAGE5_PHASE_G_EXP1_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 640,
        "max_loops": 8,
        "loop_loss_mode": "per_loop_labels",
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "seed": int(seed),
        "optimizer": "adamw",
        "learning_rate": float(os.environ.get("STAGE5_PHASE_G_EXP1_LR", "1e-5")),
        "adamw_lr": float(os.environ.get("STAGE5_PHASE_G_EXP1_LR", "1e-5")),
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": int(max_steps),
        "save_every": 0,
        "save_steps": sorted({max(1, int(max_steps) // 2)}),
        "log_every": 50,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(keeper),
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
            "start_loop": 8,
            "end_loop": 8,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "synthetic_phase": "phase_g_deterministic_gate",
        "synthetic_stage": arm,
    }
    path = run_dir / "configs" / f"{arm}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def train_arm(
    run_dir: Path,
    *,
    arm: str,
    train_jsonl: Path,
    keeper: Path,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    config = write_arm_config(run_dir, arm=arm, keeper=keeper, max_steps=max_steps, seed=seed)
    proc = run(
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
    log_path = run_dir / "train" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    checkpoint = latest_checkpoint(run_dir / "train" / arm)
    backup = backup_checkpoint_to_drive(checkpoint, run_id=run_dir.name, stage_name=arm, enabled=True)
    return {
        "arm": arm,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "train_jsonl": path_for_cli(train_jsonl),
        "config": path_for_cli(config),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_drive_backup": backup,
    }


def restore_stage_checkpoint(run_dir: Path, stage: dict[str, Any], *, arm: str) -> Path:
    expected = stage.get("checkpoint_drive_backup") or stage.get("checkpoint")
    checkpoint, metadata = restore_checkpoint(
        [expected],
        run_dir / "restored" / f"{arm}.pt",
        label=f"phase_g_{arm}",
    )
    expected_sha = stage.get("checkpoint_sha256")
    if expected_sha and metadata["selected_checkpoint_sha256"] != expected_sha:
        raise RuntimeError(
            f"Restored {arm} checkpoint SHA mismatch: "
            f"{metadata['selected_checkpoint_sha256']} != {expected_sha}"
        )
    return checkpoint


def subset_by_depth(rows: list[dict[str, Any]], rows_per_depth: int) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    subset: list[dict[str, Any]] = []
    for row in rows:
        depth = int(row["depth"])
        if counts.get(depth, 0) >= int(rows_per_depth):
            continue
        subset.append(row)
        counts[depth] = counts.get(depth, 0) + 1
    return subset


def write_mixed_training_data(data_dir: Path, *, seed: int) -> Path:
    rows = read_jsonl(data_dir / "train_injective.jsonl") + read_jsonl(data_dir / "train_abductive.jsonl")
    random.Random(int(seed)).shuffle(rows)
    path = data_dir / "train_mixed.jsonl"
    write_jsonl(path, rows)
    return path


def run_coverage_eval(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    data_jsonl: Path,
) -> dict[str, Any]:
    out_dir = run_dir / "eval" / label
    summary_path = out_dir / "summary.json"
    rows_path = out_dir / "rows.jsonl"
    run(
        [
            sys.executable,
            "eval/eval_abductive_coverage.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--sample_counts",
            "1,2,4,8,20",
            "--temperature",
            "0.7",
            "--seed",
            "2718281",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_PHASE_G_EXP1_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    return read_json(summary_path)


def assess_task(summary: dict[str, Any], *, pooled_floor: float, depth_floor: float) -> dict[str, Any]:
    pooled = float(summary["overall"]["greedy_valid_rate"])
    depth_values = {depth: float(row["greedy_valid_rate"]) for depth, row in summary["by_depth"].items()}
    minimum = min(depth_values.values()) if depth_values else 0.0
    return {
        "pooled_greedy_valid_rate": pooled,
        "min_depth_greedy_valid_rate": minimum,
        "pooled_floor": float(pooled_floor),
        "depth_floor": float(depth_floor),
        "passed": pooled >= float(pooled_floor) and minimum >= float(depth_floor),
    }


def run_guardrail(run_dir: Path, *, checkpoint: Path) -> dict[str, Any]:
    summary_path = run_dir / "guardrail" / "summary.json"
    rows_path = run_dir / "guardrail" / "rows.jsonl"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_diagonal_guardrail.py",
            "--data_jsonl",
            path_for_cli(SYNTHETIC_GUARDRAIL_DATA),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--max_depth",
            "12",
            "--value_prefix",
            "letter:",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_PHASE_G_EXP1_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    summary = read_json(summary_path)
    return {
        "active_diagonal_min": float(summary["active_diagonal_min"]),
        "floor": SYNTHETIC_FLOOR,
        "passed": float(summary["active_diagonal_min"]) >= SYNTHETIC_FLOOR,
        "summary": path_for_cli(summary_path),
    }


def main() -> int:
    run_id = os.environ.get("STAGE5_PHASE_G_EXP1_RUN_ID") or time.strftime(
        "stage5_phase_g_experiment1_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    payload = read_json(summary_path) if summary_path.exists() else {
        "kind": "stage5_phase_g_experiment1",
        "run_id": run_id,
        "status": "started",
    }
    receipt_path = Path(os.environ.get("STAGE5_PHASE_G_EXP1_RECEIPT", str(DEFAULT_RECEIPT)))
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    keeper_gate = keeper_gate_from_receipt(read_json(receipt_path))
    if keeper_gate["status"] != "green":
        raise RuntimeError(f"Locked keeper guardrail is not green: {keeper_gate}")
    keeper, keeper_restore = restore_checkpoint(
        [keeper_gate["checkpoint"]],
        run_dir / "restored" / "locked_keeper_step2000.pt",
        label="phase_g_locked_keeper_step2000",
    )
    if keeper_restore["selected_checkpoint_sha256"] != keeper_gate["checkpoint_sha256"]:
        raise RuntimeError("Locked keeper SHA does not match the preregistered receipt")
    payload["keeper_gate"] = keeper_gate

    datasets = generate_gate_data(run_dir, seed=int(os.environ.get("STAGE5_PHASE_G_EXP1_DATA_SEED", "1104729")))
    payload["datasets"] = datasets
    data_dir = run_dir / "data"
    mixed_train = write_mixed_training_data(data_dir, seed=82_001)
    for name in ("test_injective", "test_abductive"):
        smoke = subset_by_depth(read_jsonl(data_dir / f"{name}.jsonl"), rows_per_depth=16)
        write_jsonl(data_dir / f"{name}_smoke.jsonl", smoke)
    payload["status"] = "data_ready"
    write_summary(run_dir, payload)
    publish_lightweight(run_dir, message=f"Record Phase G experiment 1 data gate {run_id} [skip ci]")

    max_steps = int(os.environ.get("STAGE5_PHASE_G_EXP1_MAX_STEPS", "1000"))
    if "injective_train" not in payload:
        payload["injective_train"] = train_arm(
            run_dir,
            arm="injective_control",
            train_jsonl=data_dir / "train_injective.jsonl",
            keeper=keeper,
            max_steps=max_steps,
            seed=81_001,
        )
        payload["status"] = "injective_trained"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record Phase G injective training {run_id} [skip ci]")
    injective_checkpoint = restore_stage_checkpoint(
        run_dir, payload["injective_train"], arm="injective_control"
    )
    if "injective_smoke" not in payload:
        payload["injective_smoke"] = run_coverage_eval(
            run_dir,
            label="injective_smoke",
            checkpoint=injective_checkpoint,
            data_jsonl=data_dir / "test_injective_smoke.jsonl",
        )
        write_summary(run_dir, payload)
    if float(payload["injective_smoke"]["overall"]["greedy_valid_rate"]) < 0.50:
        payload["status"] = "blocked_injective_smoke"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record blocked Phase G injective smoke {run_id} [skip ci]")
        return 2
    if "injective_full" not in payload:
        payload["injective_full"] = run_coverage_eval(
            run_dir,
            label="injective_full",
            checkpoint=injective_checkpoint,
            data_jsonl=data_dir / "test_injective.jsonl",
        )
        payload["injective_gate"] = assess_task(
            payload["injective_full"],
            pooled_floor=INJECTIVE_POOLED_FLOOR,
            depth_floor=INJECTIVE_DEPTH_FLOOR,
        )
        payload["status"] = "injective_evaluated"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record Phase G injective evaluation {run_id} [skip ci]")
    if not payload["injective_gate"]["passed"]:
        payload["status"] = "blocked_injective_gate"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record blocked Phase G injective gate {run_id} [skip ci]")
        return 2

    if "abductive_train" not in payload:
        payload["abductive_train"] = train_arm(
            run_dir,
            arm="abductive_mixed",
            train_jsonl=mixed_train,
            keeper=keeper,
            max_steps=max_steps,
            seed=82_001,
        )
        payload["status"] = "abductive_trained"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record Phase G abductive training {run_id} [skip ci]")
    abductive_checkpoint = restore_stage_checkpoint(
        run_dir, payload["abductive_train"], arm="abductive_mixed"
    )
    if "abductive_smoke" not in payload:
        payload["abductive_smoke"] = run_coverage_eval(
            run_dir,
            label="abductive_smoke",
            checkpoint=abductive_checkpoint,
            data_jsonl=data_dir / "test_abductive_smoke.jsonl",
        )
        write_summary(run_dir, payload)
    if float(payload["abductive_smoke"]["overall"]["greedy_valid_rate"]) < 0.40:
        payload["status"] = "blocked_abductive_smoke"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record blocked Phase G abductive smoke {run_id} [skip ci]")
        return 2
    if "abductive_full" not in payload:
        payload["abductive_full"] = run_coverage_eval(
            run_dir,
            label="abductive_full",
            checkpoint=abductive_checkpoint,
            data_jsonl=data_dir / "test_abductive.jsonl",
        )
        payload["abductive_gate"] = assess_task(
            payload["abductive_full"],
            pooled_floor=ABDUCTIVE_POOLED_FLOOR,
            depth_floor=ABDUCTIVE_DEPTH_FLOOR,
        )
        payload["status"] = "abductive_evaluated"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record Phase G abductive evaluation {run_id} [skip ci]")
    if not payload["abductive_gate"]["passed"]:
        payload["status"] = "blocked_abductive_gate"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record blocked Phase G abductive gate {run_id} [skip ci]")
        return 2

    if "synthetic_guardrail" not in payload:
        payload["synthetic_guardrail"] = run_guardrail(run_dir, checkpoint=abductive_checkpoint)
        payload["status"] = "guardrail_evaluated"
        write_summary(run_dir, payload)
        publish_lightweight(run_dir, message=f"Record Phase G synthetic guardrail {run_id} [skip ci]")

    payload["phase_g_alpha_training_unlocked"] = bool(
        payload["injective_gate"]["passed"]
        and payload["abductive_gate"]["passed"]
        and payload["synthetic_guardrail"]["passed"]
    )
    payload["status"] = (
        "experiment1_passed" if payload["phase_g_alpha_training_unlocked"] else "blocked_synthetic_guardrail"
    )
    write_summary(run_dir, payload)
    publish_lightweight(run_dir, message=f"Finish Phase G experiment 1 {run_id} [skip ci]")
    return 0 if payload["phase_g_alpha_training_unlocked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
