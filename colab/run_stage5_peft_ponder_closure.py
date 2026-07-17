"""Run the corrected-loop LoRA installation ladder and optional Ponder controller."""

from __future__ import annotations

import hashlib
import json
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

from colab.run_stage5_natural_surface_receipts import materialize_datasets
from colab.stage5_n24_rung import tier1_canary_verdict
from colab.stage5_publish_utils import publishable_artifact_paths
from training.peft_ponder_closure import (
    ARMS,
    build_base_capability_canary_rows,
    canary_baseline_gate,
    full_block_comparison,
    historical_archive_receipt,
    locked_spec,
    next_p1_action,
    p1_gate,
    p2_gate,
)


REFERENCE_ROOT = ROOT / "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827"
REFERENCE_SUMMARY = REFERENCE_ROOT / "summary.json"
TRAIN_LE2 = REFERENCE_ROOT / "data/train_chain_symbol_depth_le2_sft.jsonl"
TRAIN_LE4 = REFERENCE_ROOT / "data/train_chain_symbol_depth_le4_sft.jsonl"
HELDOUT64 = REFERENCE_ROOT / "data/test_chain_mcq_heldout64.jsonl"
DRIVE_CHECKPOINT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints")


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str | os.PathLike[str]],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        list(map(str, command)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(chunks), None)
    if check and result.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(result.stdout.splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Could not parse checkpoint step from {path}")
    return int(match.group(1))


def latest_checkpoint(path: Path) -> Path:
    candidates = sorted(path.glob("*_step_*.pt"), key=checkpoint_step)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    return candidates[-1]


def balanced_prefix(rows: list[dict[str, Any]], *, rows_per_depth: int, max_depth: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for row in rows:
        depth = int(row["depth"])
        if depth > max_depth or counts.get(depth, 0) >= rows_per_depth:
            continue
        out.append(row)
        counts[depth] = counts.get(depth, 0) + 1
    expected = {depth: rows_per_depth for depth in range(1, max_depth + 1)}
    if counts != expected:
        raise RuntimeError(f"Unbalanced frozen prefix: {counts} != {expected}")
    return out


def prepare_data(run_dir: Path) -> dict[str, Any]:
    required = [REFERENCE_SUMMARY, TRAIN_LE2, TRAIN_LE4, HELDOUT64]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing locked corrected-chain artifacts: {missing}")
    natural_dir = run_dir / "natural_source"
    natural = materialize_datasets(natural_dir, seed=20260709)
    relay = read_jsonl(ROOT / natural["files"]["paired_relay_d1_12"])
    pointer = read_jsonl(ROOT / natural["files"]["paired_pointer_d1_12"])
    full = balanced_prefix(relay, rows_per_depth=16, max_depth=8)
    full.extend(balanced_prefix(pointer, rows_per_depth=16, max_depth=8))
    canary_path = run_dir / "data/base_capability_canary_64.jsonl"
    full_path = run_dir / "data/natural_loop1_full_16_each_d1_8.jsonl"
    write_jsonl(canary_path, build_base_capability_canary_rows())
    write_jsonl(full_path, full)
    return {
        "reference_summary": path_for_cli(REFERENCE_SUMMARY),
        "train_depth_le2": path_for_cli(TRAIN_LE2),
        "train_depth_le4": path_for_cli(TRAIN_LE4),
        "heldout64": path_for_cli(HELDOUT64),
        "sha256": {
            "train_depth_le2": sha256_file(TRAIN_LE2),
            "train_depth_le4": sha256_file(TRAIN_LE4),
            "heldout64": sha256_file(HELDOUT64),
        },
        "base_capability_canary": path_for_cli(canary_path),
        "natural_full": path_for_cli(full_path),
    }


def write_config(
    path: Path,
    *,
    arm_name: str,
    rank: int,
    alpha: int,
    stage_name: str,
    train_jsonl: Path,
    output_dir: Path,
    max_loops: int,
    max_steps: int,
    resume_from: Path | None,
    controller_only: bool = False,
    canary_jsonl: Path | None = None,
    canary_baseline_accuracy: float | None = None,
) -> None:
    learning_rate = float(os.environ.get("STAGE5_PEFT_LR", "1e-5"))
    cfg: dict[str, Any] = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("STAGE5_PEFT_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 512,
        "max_loops": max_loops,
        "loop_loss_mode": "halting_weighted" if controller_only else "per_loop_labels",
        "initial_halt_prob": 0.15,
        "beta": float(os.environ.get("STAGE5_PONDER_BETA", "0.02")) if controller_only else 0.0,
        "halt_target_nll_weight": (
            float(os.environ.get("STAGE5_PONDER_TARGET_NLL_WEIGHT", "0.1"))
            if controller_only
            else 0.0
        ),
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "minimum_effective_batch_size": 1,
        "seed": int(os.environ.get("STAGE5_PEFT_SEED", "20260717")),
        "optimizer": "adamw",
        "reject_muon": True,
        "learning_rate": (
            float(os.environ.get("STAGE5_PONDER_LR", "1e-4"))
            if controller_only
            else learning_rate
        ),
        "adamw_lr": learning_rate,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": max_steps,
        "save_every": 1000,
        "log_every": 25 if controller_only else 100,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 1.0 if controller_only else 10.0,
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "training_mode": "controller_only" if controller_only else "frozen_lora",
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "require_active_supervision": True,
        "require_nonzero_train_gradient": True,
        "require_frozen_base_hash": True,
        "output_dir": path_for_cli(output_dir),
        "resume_lora": {
            "enabled": True,
            "rank": rank,
            "alpha": alpha,
            "dropout": 0.0,
        },
        "merge_lora_before_unfreeze": False,
        "require_lora_loaded_before_merge": False,
        "train_auxiliary": {
            "bridge": not controller_only,
            "halting": controller_only,
            "reentry_adapter": False,
            "latent": False,
        },
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": max_loops,
            "end_loop": max_loops,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "checkpoint_include_frozen_prefixes": ["bridge."] if controller_only else [],
        "checkpoint_include_frozen_lora": controller_only,
        "synthetic_phase": "peft_ponder_closure",
        "synthetic_stage": stage_name,
        "peft_arm": arm_name,
    }
    if canary_jsonl is not None and canary_baseline_accuracy is not None:
        cfg.update(
            {
                "canary_every": 1000,
                "canary_jsonl": path_for_cli(canary_jsonl),
                "canary_value_prefix": "name:",
                "canary_baseline_accuracy": float(canary_baseline_accuracy),
                "canary_hard_stop_delta": -0.03,
            }
        )
    if resume_from is not None:
        cfg["resume_from"] = path_for_cli(resume_from)
        cfg["require_all_lora_loaded"] = True
        cfg["require_loaded_prefixes"] = ["bridge."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def train_stage(
    run_dir: Path,
    *,
    arm_name: str,
    rank: int,
    alpha: int,
    stage_name: str,
    train_jsonl: Path,
    max_loops: int,
    max_steps: int,
    resume_from: Path | None,
    controller_only: bool = False,
    canary_jsonl: Path | None = None,
    canary_baseline_accuracy: float | None = None,
) -> dict[str, Any]:
    output_dir = run_dir / "train" / arm_name / stage_name
    config = run_dir / "configs" / arm_name / f"{stage_name}.yaml"
    write_config(
        config,
        arm_name=arm_name,
        rank=rank,
        alpha=alpha,
        stage_name=stage_name,
        train_jsonl=train_jsonl,
        output_dir=output_dir,
        max_loops=max_loops,
        max_steps=max_steps,
        resume_from=resume_from,
        controller_only=controller_only,
        canary_jsonl=canary_jsonl,
        canary_baseline_accuracy=canary_baseline_accuracy,
    )
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
    log_path = output_dir / "train.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    summary_path = output_dir / "train_unfrozen_recurrent_summary.json"
    training_payload = read_json(summary_path)
    return {
        "name": stage_name,
        "config": path_for_cli(config),
        "output_dir": path_for_cli(output_dir),
        "train_log": path_for_cli(log_path),
        "training_summary": path_for_cli(summary_path),
        "checkpoint": path_for_cli(latest_checkpoint(output_dir)),
        "interval_checkpoints": [
            path_for_cli(path)
            for path in sorted(output_dir.glob("*_step_*.pt"), key=checkpoint_step)
        ],
        "base_hash": {
            "start": training_payload.get("pretrained_base_sha256_start"),
            "end": training_payload.get("pretrained_base_sha256_end"),
            "unchanged": training_payload.get("pretrained_base_hash_unchanged"),
        },
        "status": training_payload.get("status"),
        "canary_trace": training_payload.get("canary_trace", []),
        "trainable_parameters": training_payload.get("trainable_parameters", {}),
    }


def eval_active(
    run_dir: Path,
    *,
    label: str,
    checkpoint: Path,
    data_jsonl: Path,
    rank: int,
    alpha: int,
    loop_counts: str,
    value_prefix: str = "letter:",
) -> dict[str, Any]:
    out_dir = run_dir / "eval" / label
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--loop_counts",
            loop_counts,
            "--threshold",
            "0.71",
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--split",
            "6,18",
            "--bridge_projection_mode",
            "split",
            "--lora_rank",
            str(rank),
            "--lora_alpha",
            str(alpha),
            "--dtype",
            os.environ.get("STAGE5_PEFT_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--progress_every",
            "64",
        ]
    )
    return {
        "summary": path_for_cli(summary_path),
        "payload": read_json(summary_path),
    }


def run_identity(run_dir: Path, *, arm_name: str, rank: int, alpha: int) -> dict[str, Any]:
    summary_path = run_dir / "identity" / arm_name / "summary.json"
    checkpoint = run_dir / "identity" / arm_name / "peft_identity_step_0.pt"
    run(
        [
            sys.executable,
            "eval/eval_peft_identity.py",
            "--rank",
            str(rank),
            "--alpha",
            str(alpha),
            "--dtype",
            os.environ.get("STAGE5_PEFT_DTYPE", "bfloat16"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--output_summary",
            path_for_cli(summary_path),
            "--output_checkpoint",
            path_for_cli(checkpoint),
        ]
    )
    return {**read_json(summary_path), "summary": path_for_cli(summary_path)}


def backup_checkpoint(checkpoint: Path, *, run_id: str, label: str) -> dict[str, Any]:
    destination = DRIVE_CHECKPOINT_ROOT / run_id / label / checkpoint.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def publish(run_dir: Path, message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        if path.suffix == ".pt" or "natural_source" in path.parts or path.name == "rows.jsonl":
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    pushed = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if pushed.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# PEFT + Ponder Closure - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Next action: `{payload.get('next_action')}`",
        f"- Historical repaired-loop PEFT arm found: `{payload['historical_archive']['repaired_loop_peft_arm_found']}`",
        "",
        "## P1",
        "",
        "| Arm | Rank | Steps | Gate | Depth counts | Base hash |",
        "|---|---:|---:|---|---|---|",
    ]
    for arm in payload.get("p1_results", []):
        lines.append(
            f"| {arm['arm']} | {arm['rank']} | {arm['total_steps']} | "
            f"{arm.get('gate', {}).get('passed')} | {arm.get('gate', {}).get('counts')} | "
            f"{arm.get('base_hash_unchanged')} |"
        )
    if payload.get("p2"):
        lines.extend(["", "## P2", "", f"- Gate: `{payload['p2'].get('gate')}`"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def evaluate_arm(
    run_dir: Path,
    *,
    arm_name: str,
    rank: int,
    alpha: int,
    stages: list[dict[str, Any]],
    data: dict[str, Any],
    identity: dict[str, Any],
    total_steps: int,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    intervals: list[dict[str, Any]] = []
    baseline_accuracy = float(baseline["payload"]["active_total"]["accuracy"])
    expected_base_hash = str(identity["pretrained_base_sha256"])
    stage_base_hashes = [
        str(stage["base_hash"].get("start")) for stage in stages
    ] + [
        str(stage["base_hash"].get("end")) for stage in stages
    ]
    if any(value != expected_base_hash for value in stage_base_hashes):
        raise RuntimeError(
            f"{arm_name} base initialization/hash mismatch: "
            f"expected={expected_base_hash}, observed={stage_base_hashes}"
        )
    cumulative_offset = 0
    hard_stop = False
    for stage in stages:
        stage_canaries = {
            int(row["step"]): row for row in stage.get("canary_trace", [])
        }
        for checkpoint_text in stage["interval_checkpoints"]:
            checkpoint = ROOT / checkpoint_text
            local_step = checkpoint_step(checkpoint)
            cumulative_step = cumulative_offset + local_step
            gate_eval = eval_active(
                run_dir,
                label=f"{arm_name}_dose_{cumulative_step}",
                checkpoint=checkpoint,
                data_jsonl=HELDOUT64,
                rank=rank,
                alpha=alpha,
                loop_counts="1,2,3,4",
            )
            canary_receipt = stage_canaries.get(local_step)
            if canary_receipt is None:
                raise RuntimeError(
                    f"Missing in-training Tier-1 canary at {arm_name} local step {local_step}"
                )
            candidate_accuracy = float(canary_receipt["accuracy"])
            verdict = tier1_canary_verdict(
                accuracy_delta=candidate_accuracy - baseline_accuracy,
                ppl_relative_delta=None,
            )
            intervals.append(
                {
                    "cumulative_step": cumulative_step,
                    "checkpoint": checkpoint_text,
                    "active_summary": gate_eval["summary"],
                    "gate": p1_gate(gate_eval["payload"]),
                    "tier1": {
                        "baseline_accuracy": baseline_accuracy,
                        "candidate_accuracy": candidate_accuracy,
                        "verdict": verdict,
                        "in_training_receipt": canary_receipt,
                    },
                }
            )
            if verdict["status"] == "red_hard_stop":
                hard_stop = True
                break
        if hard_stop:
            break
        cumulative_offset += int(read_json(ROOT / stage["training_summary"])["final_step"])
    final_checkpoint = ROOT / stages[-1]["checkpoint"]
    final_eval = eval_active(
        run_dir,
        label=f"{arm_name}_final_heldout64",
        checkpoint=final_checkpoint,
        data_jsonl=HELDOUT64,
        rank=rank,
        alpha=alpha,
        loop_counts="1,2,3,4",
    )
    final_natural = eval_active(
        run_dir,
        label=f"{arm_name}_final_natural_loop1",
        checkpoint=final_checkpoint,
        data_jsonl=ROOT / data["natural_full"],
        rank=rank,
        alpha=alpha,
        loop_counts="1",
        value_prefix="name:",
    )
    gate = p1_gate(final_eval["payload"])
    trainer_hard_stop = any(stage.get("status") == "hard_stopped_canary" for stage in stages)
    if hard_stop or trainer_hard_stop:
        gate["passed"] = False
        gate["blocked_by_tier1"] = True
    backup = backup_checkpoint(final_checkpoint, run_id=run_dir.name, label=arm_name)
    return {
        "arm": arm_name,
        "rank": rank,
        "alpha": alpha,
        "total_steps": total_steps,
        "identity": identity,
        "tier1_baseline": {
            "accuracy": baseline_accuracy,
            "gate": canary_baseline_gate(baseline["payload"]),
            "summary": baseline["summary"],
        },
        "stages": stages,
        "intervals": intervals,
        "gate": gate,
        "full_block_comparison": full_block_comparison(final_eval["payload"]),
        "final_active_summary": final_eval["summary"],
        "final_natural_loop1_summary": final_natural["summary"],
        "checkpoint": path_for_cli(final_checkpoint),
        "checkpoint_backup": backup,
        "base_hash_unchanged": all(
            bool(stage["base_hash"].get("unchanged")) for stage in stages
        ),
    }


def run_p1_arm(run_dir: Path, arm: Any, data: dict[str, Any]) -> dict[str, Any]:
    identity = run_identity(run_dir, arm_name=arm.name, rank=arm.rank, alpha=arm.alpha)
    if not identity["passed"]:
        raise RuntimeError(f"Identity gate failed for {arm.name}: {identity}")
    baseline = eval_active(
        run_dir,
        label=f"{arm.name}_base_capability_canary_baseline",
        checkpoint=ROOT / identity["checkpoint"],
        data_jsonl=ROOT / data["base_capability_canary"],
        rank=arm.rank,
        alpha=arm.alpha,
        loop_counts="1",
        value_prefix="name:",
    )
    baseline_gate = canary_baseline_gate(baseline["payload"])
    if not baseline_gate["passed"]:
        raise RuntimeError(
            "Base capability canary is vacuous; refusing to train: "
            f"{baseline_gate}"
        )
    baseline_accuracy = float(baseline["payload"]["active_total"]["accuracy"])
    stage12 = train_stage(
        run_dir,
        arm_name=arm.name,
        rank=arm.rank,
        alpha=arm.alpha,
        stage_name="depth_le2",
        train_jsonl=TRAIN_LE2,
        max_loops=2,
        max_steps=2000,
        resume_from=None,
        canary_jsonl=ROOT / data["base_capability_canary"],
        canary_baseline_accuracy=baseline_accuracy,
    )
    if stage12["status"] == "hard_stopped_canary":
        return evaluate_arm(
            run_dir,
            arm_name=arm.name,
            rank=arm.rank,
            alpha=arm.alpha,
            stages=[stage12],
            data=data,
            identity=identity,
            total_steps=int(read_json(ROOT / stage12["training_summary"])["final_step"]),
            baseline=baseline,
        )
    stage4 = train_stage(
        run_dir,
        arm_name=arm.name,
        rank=arm.rank,
        alpha=arm.alpha,
        stage_name="depth_le4",
        train_jsonl=TRAIN_LE4,
        max_loops=4,
        max_steps=4000,
        resume_from=ROOT / stage12["checkpoint"],
        canary_jsonl=ROOT / data["base_capability_canary"],
        canary_baseline_accuracy=baseline_accuracy,
    )
    result = evaluate_arm(
        run_dir,
        arm_name=arm.name,
        rank=arm.rank,
        alpha=arm.alpha,
        stages=[stage12, stage4],
        data=data,
        identity=identity,
        total_steps=6000,
        baseline=baseline,
    )
    if arm.name == "R256" and not result["gate"]["passed"] and not result["gate"].get("blocked_by_tier1"):
        rider = train_stage(
            run_dir,
            arm_name=arm.name,
            rank=arm.rank,
            alpha=arm.alpha,
            stage_name="depth_le4_rider",
            train_jsonl=TRAIN_LE4,
            max_loops=4,
            max_steps=6000,
            resume_from=ROOT / stage4["checkpoint"],
            canary_jsonl=ROOT / data["base_capability_canary"],
            canary_baseline_accuracy=baseline_accuracy,
        )
        result = evaluate_arm(
            run_dir,
            arm_name=arm.name,
            rank=arm.rank,
            alpha=arm.alpha,
            stages=[stage12, stage4, rider],
            data=data,
            identity=identity,
            total_steps=12000,
            baseline=baseline,
        )
    return result


def run_p2(
    run_dir: Path,
    arm_result: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    arm_name = str(arm_result["arm"])
    rank = int(arm_result["rank"])
    alpha = int(arm_result["alpha"])
    stage = train_stage(
        run_dir,
        arm_name=arm_name,
        rank=rank,
        alpha=alpha,
        stage_name="ponder_controller",
        train_jsonl=TRAIN_LE4,
        max_loops=4,
        max_steps=2000,
        resume_from=ROOT / arm_result["checkpoint"],
        controller_only=True,
    )
    checkpoint = ROOT / stage["checkpoint"]
    eval_dir = run_dir / "eval" / f"{arm_name}_ponder"
    run(
        [
            sys.executable,
            "eval/eval_ponder_depth.py",
            "--data_jsonl",
            path_for_cli(HELDOUT64),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(eval_dir / "rows.jsonl"),
            "--output_summary",
            path_for_cli(eval_dir / "summary.json"),
            "--lora_rank",
            str(rank),
            "--lora_alpha",
            str(alpha),
            "--dtype",
            os.environ.get("STAGE5_PEFT_DTYPE", "bfloat16"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    training_summary = read_json(ROOT / stage["training_summary"])
    expected_base_hash = str(arm_result["identity"]["pretrained_base_sha256"])
    if str(stage["base_hash"].get("start")) != expected_base_hash or str(
        stage["base_hash"].get("end")
    ) != expected_base_hash:
        raise RuntimeError(
            "Ponder controller did not preserve the passing arm's base hash"
        )
    trainable = stage.get("trainable_parameters") or {}
    if int(trainable.get("recurrent_block", -1)) != 0 or int(trainable.get("bridge", -1)) != 0:
        raise RuntimeError(f"Ponder phase unfroze mechanism parameters: {trainable}")
    eval_summary = read_json(eval_dir / "summary.json")
    final_natural = eval_active(
        run_dir,
        label=f"{arm_name}_ponder_final_natural_loop1",
        checkpoint=checkpoint,
        data_jsonl=ROOT / data["natural_full"],
        rank=rank,
        alpha=alpha,
        loop_counts="1",
        value_prefix="name:",
    )
    return {
        "arm": arm_name,
        "stage": stage,
        "eval_summary": path_for_cli(eval_dir / "summary.json"),
        "final_natural_loop1_summary": final_natural["summary"],
        "gate": p2_gate(training_summary, eval_summary),
        "checkpoint_backup": backup_checkpoint(
            checkpoint,
            run_id=run_dir.name,
            label=f"{arm_name}_ponder",
        ),
    }


def main() -> int:
    run_id = os.environ.get("STAGE5_PEFT_PONDER_RUN_ID") or time.strftime(
        "stage5_peft_ponder_closure_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": "stage5_peft_ponder_closure",
        "run_id": run_id,
        "status": "started",
        "locked_spec": locked_spec(),
        "historical_archive": historical_archive_receipt(),
        "invalidated_predecessor": {
            "run_id": "stage5_peft_ponder_closure_20260717_175439",
            "status": "invalidated_before_first_checkpoint",
            "reason": (
                "The natural iterative-function Tier-1 baseline was 0/32, "
                "making the -3pp preservation hard stop vacuous."
            ),
        },
        "data": None,
        "p1_results": [],
        "p2": None,
        "next_action": "prepare_data",
    }
    write_summary(run_dir, payload)
    payload["data"] = prepare_data(run_dir)
    payload["status"] = "data_ready"
    payload["next_action"] = "run_R16"
    write_summary(run_dir, payload)
    publish(run_dir, f"Record PEFT closure preregistration {run_id} [skip ci]")

    for arm in ARMS:
        expected = f"run_{arm.name}"
        action = next_p1_action(payload["p1_results"])
        if action != expected:
            break
        result = run_p1_arm(run_dir, arm, payload["data"])
        payload["p1_results"].append(result)
        payload["next_action"] = next_p1_action(payload["p1_results"])
        payload["status"] = f"{arm.name}_{'passed' if result['gate']['passed'] else 'failed'}"
        write_summary(run_dir, payload)
        publish(run_dir, f"Record PEFT closure {arm.name} {run_id} [skip ci]")
        if result["gate"].get("blocked_by_tier1"):
            payload["next_action"] = "strategy_review"
            break
        if result["gate"]["passed"]:
            break

    if payload["next_action"] == "run_P2_on_first_pass":
        passing = next(result for result in payload["p1_results"] if result["gate"]["passed"])
        payload["p2"] = run_p2(run_dir, passing, payload["data"])
        payload["status"] = "finished_p2_passed" if payload["p2"]["gate"]["passed"] else "finished_p2_failed"
        payload["next_action"] = "strategy_review"
    elif payload["next_action"] == "close_P1_bounded_refutation":
        payload["status"] = "finished_p1_bounded_refutation"
        payload["next_action"] = "strategy_review"
    else:
        payload["status"] = "paused_by_guardrail"
        payload["next_action"] = "strategy_review"
    write_summary(run_dir, payload)
    publish(run_dir, f"Finish PEFT + Ponder closure {run_id} [skip ci]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
