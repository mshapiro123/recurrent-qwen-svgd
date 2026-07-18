"""Run the preregistered R16 adapter-budget Arm E from the fresh base surgery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.stage5_publish_utils import publishable_artifact_paths
from training.adapter_budget_arm import (
    locked_spec,
    normalized_text_sha256,
    score_adapter_budget_arm,
)
from training.peft_ponder_closure import build_base_capability_canary_rows, canary_baseline_gate


RUN_ID = os.environ.get(
    "STAGE5_ADAPTER_BUDGET_RUN_ID",
    "stage5_adapter_budget_arm_e_20260718",
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_ADAPTER_BUDGET_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/{RUN_ID}",
    )
)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("STAGE5_ADAPTER_BUDGET_DTYPE", "bfloat16")
RANK = 16
ALPHA = 32
LOCKED_ADAPTER = {"rank": 16, "alpha": 32}
EXPECTED_FORWARD_ACTIVE = 6_007_425
EXPECTED_PRETRAINED_BASE_SHA256 = (
    "960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f"
)
EXPECTED_IMMUTABLE_SHA256 = {
    "chain_depth_le2": "56fbb7774d5716632b66b60bad3d0067c9f92f58f0db8aa7297eb637116d1b69",
    "chain_depth_le4": "829cbd3d381329470e8621094e7c5326f612552a56f38b3b09d52d873603f636",
    "chain_depth_le8": "cf61c14c2629f2caa7e1b6bd100adb122a468d5285b74970aaa4aebfbb56fd12",
    "chain_depth_le8_dose": "cf61c14c2629f2caa7e1b6bd100adb122a468d5285b74970aaa4aebfbb56fd12",
    "frozen_eval": "3de844669aba303063e6932f5852914ee0993e531c8e65c2a4c4b18e219b3fc8",
    "arm_a_rows": "0b582b9e49a59eff4a9b739e0df6520328de77f0d48abe8892ff0fea78817e65",
}
IMMUTABLE_DATASET_HASH_MODE = "utf8_lf_normalized"

CHAIN_ROOT = ROOT / "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827"
DEPTH8_ROOT = ROOT / "outputs/stage5/stage5_depth_support_ladder8_20260705_204923"
FROZEN_EVAL = (
    ROOT
    / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"
)
ARM_A_ROWS = (
    ROOT
    / "outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/eval/same_reader_final_rows.jsonl"
)


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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
        print("\n".join(result.stdout.splitlines()[-200:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step: {path}")
    return int(match.group(1))


def publish(message: str) -> None:
    subprocess.run(
        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
        cwd=ROOT,
        check=False,
    )
    for path in publishable_artifact_paths(RUN_DIR):
        if path.suffix == ".pt" or path.name.endswith("_rows.jsonl"):
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    pushed = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if pushed.returncode:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def copy_checkpoint_to_drive(checkpoint: Path, stage_name: str) -> dict[str, Any]:
    destination = DRIVE_ROOT / stage_name / checkpoint.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def restore_checkpoint_from_drive(stage_name: str, expected: Path) -> bool:
    source = DRIVE_ROOT / stage_name / expected.name
    if not source.exists():
        return False
    expected.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, expected)
    if sha256_file(source) != sha256_file(expected):
        raise RuntimeError(f"Drive restore SHA mismatch for {stage_name}: {source}")
    print(f"restored_stage_checkpoint={expected} source={source}", flush=True)
    return True


def generate_locked_data() -> dict[str, Any]:
    primitive_dir = RUN_DIR / "data" / "primitive_depth1_seed20260701"
    primitive_summary = primitive_dir / "summary.json"
    if not primitive_summary.exists():
        run(
            [
                sys.executable,
                "training/generate_synthetic_depth_task.py",
                "--output_dir",
                path_for_cli(primitive_dir),
                "--n_symbols",
                "16",
                "--max_depth",
                "1",
                "--rows_per_depth",
                "256",
                "--seed",
                "20260701",
                "--num_choices",
                "4",
                "--max_target_loops",
                "1",
            ]
        )
    primitive = read_json(primitive_summary)
    expected_primitive = {
        "n_symbols": 16,
        "max_depth": 1,
        "rows_per_depth": 256,
        "seed": 20260701,
        "num_choices": 4,
        "max_target_loops": 1,
        "value_prefix": "",
    }
    if primitive.get("config") != expected_primitive:
        raise RuntimeError(
            f"Primitive data config drifted: {primitive.get('config')} != {expected_primitive}"
        )

    chain2 = CHAIN_ROOT / "data/train_chain_symbol_depth_le2_sft.jsonl"
    chain4 = CHAIN_ROOT / "data/train_chain_symbol_depth_le4_sft.jsonl"
    chain8 = DEPTH8_ROOT / "data/train_chain_symbol_sft.jsonl"
    required = [chain2, chain4, chain8, FROZEN_EVAL, ARM_A_ROWS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing immutable Arm A artifacts: {missing}")

    frozen_rows = read_jsonl(FROZEN_EVAL)
    smoke_paths: dict[str, str] = {}
    for depth, rows_per_depth in ((2, 64), (4, 32), (8, 16)):
        counts = {value: 0 for value in range(1, depth + 1)}
        selected: list[dict[str, Any]] = []
        for row in frozen_rows:
            row_depth = int(row["depth"])
            if row_depth <= depth and counts[row_depth] < rows_per_depth:
                selected.append(row)
                counts[row_depth] += 1
        if counts != {value: rows_per_depth for value in range(1, depth + 1)}:
            raise RuntimeError(f"Could not build balanced depth-{depth} smoke: {counts}")
        path = RUN_DIR / "data" / f"phase_a_smoke_128_depth_le{depth}.jsonl"
        write_jsonl(path, selected)
        smoke_paths[str(depth)] = path_for_cli(path)

    canary = RUN_DIR / "data/base_capability_canary_64.jsonl"
    write_jsonl(canary, build_base_capability_canary_rows())
    files = {
        "primitive_depth1": primitive_dir / "train_mcq_option_text_sft.jsonl",
        "chain_depth_le2": chain2,
        "chain_depth_le4": chain4,
        "chain_depth_le8": chain8,
        "chain_depth_le8_dose": chain8,
        "frozen_eval": FROZEN_EVAL,
        "arm_a_rows": ARM_A_ROWS,
        "canary": canary,
    }
    observed_sha256 = {
        name: normalized_text_sha256(path) for name, path in files.items()
    }
    for name, expected_sha in EXPECTED_IMMUTABLE_SHA256.items():
        if observed_sha256[name] != expected_sha:
            raise RuntimeError(
                f"Immutable Arm A artifact hash drifted for {name}: "
                f"observed={observed_sha256[name]} expected={expected_sha}"
            )
    return {
        "files": {name: path_for_cli(path) for name, path in files.items()},
        "sha256": observed_sha256,
        "sha256_mode": IMMUTABLE_DATASET_HASH_MODE,
        "smoke_paths": smoke_paths,
        "primitive_summary": path_for_cli(primitive_summary),
    }


def write_train_config(
    *,
    stage: dict[str, Any],
    train_jsonl: Path,
    output_dir: Path,
    resume_from: Path,
    canary_path: Path,
    canary_baseline: float,
) -> Path:
    config_path = RUN_DIR / "configs" / f"{stage['name']}.yaml"
    max_loops = int(stage["max_loops"])
    chain_stage = stage["supervision"] == "per_loop_labels"
    config: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 512,
        "max_loops": max_loops,
        "loop_loss_mode": "per_loop_labels" if chain_stage else "halting_weighted",
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "batch_size": int(stage["batch_size"]),
        "gradient_accumulation_steps": int(stage["gradient_accumulation_steps"]),
        "minimum_effective_batch_size": 1,
        # Historical Arm A configs omitted the optimizer RNG seed, whose
        # trainer default is zero. Dataset generation seeds are recorded
        # separately in the locked protocol.
        "seed": int(stage["training_seed"]),
        "optimizer": "adamw",
        "reject_muon": True,
        "learning_rate": float(stage["learning_rate"]),
        "adamw_lr": float(stage["learning_rate"]),
        "weight_decay": float(stage["weight_decay"]),
        "max_grad_norm": float(stage["max_grad_norm"]),
        "max_steps": int(stage["max_steps"]),
        "save_every": 1000,
        "save_steps": [],
        "log_every": 100,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": float(stage["bridge_prelude_lr_multiplier"]),
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "training_mode": "frozen_lora",
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "require_active_supervision": True,
        "require_nonzero_train_gradient": True,
        "require_frozen_base_hash": True,
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(resume_from),
        "resume_lora": {
            "enabled": True,
            "rank": RANK,
            "alpha": ALPHA,
            "dropout": 0.0,
        },
        "require_all_lora_loaded": True,
        "require_loaded_prefixes": ["bridge."],
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
            "start_loop": max_loops,
            "end_loop": max_loops,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "track_loop_dose": chain_stage,
        "dose_assert_every": 1000,
        "canary_every": 1000,
        "canary_jsonl": path_for_cli(canary_path),
        "canary_value_prefix": "name:",
        "canary_baseline_accuracy": canary_baseline,
        "canary_hard_stop_delta": -0.03,
        "checkpoint_backup_every": 1000,
        "checkpoint_backup_dir": str(DRIVE_ROOT / str(stage["name"])),
        "progress_backup_path": str(DRIVE_ROOT / str(stage["name"]) / "progress.json"),
        "synthetic_phase": "adapter_budget_depth_profile",
        "synthetic_stage": str(stage["name"]),
        "adapter_budget_arm": "E",
        "adapter_budget_rank": RANK,
        "adapter_budget_alpha": ALPHA,
        "fresh_base_qwen_surgery": True,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_identity() -> tuple[Path, dict[str, Any]]:
    output_dir = RUN_DIR / "identity"
    summary = output_dir / "summary.json"
    checkpoint = output_dir / "peft_identity_step_0.pt"
    if not summary.exists() or not checkpoint.exists():
        run(
            [
                sys.executable,
                "eval/eval_peft_identity.py",
                "--model_name",
                MODEL_NAME,
                "--rank",
                str(RANK),
                "--alpha",
                str(ALPHA),
                "--threshold",
                "0.001",
                "--dtype",
                DTYPE,
                "--adapter_dtype",
                "float32",
                "--device",
                DEVICE,
                "--output_summary",
                path_for_cli(summary),
                "--output_checkpoint",
                path_for_cli(checkpoint),
            ]
        )
    payload = read_json(summary)
    if not payload.get("passed"):
        raise RuntimeError(f"Step-zero identity gate failed: {payload}")
    if payload.get("pretrained_base_sha256") != EXPECTED_PRETRAINED_BASE_SHA256:
        raise RuntimeError(
            "Fresh-base surgery hash mismatch: "
            f"observed={payload.get('pretrained_base_sha256')} "
            f"expected={EXPECTED_PRETRAINED_BASE_SHA256}"
        )
    print(
        f"[assert-ok] fresh_base_sha256={EXPECTED_PRETRAINED_BASE_SHA256}",
        flush=True,
    )
    return checkpoint, payload


def eval_active(
    *,
    label: str,
    checkpoint: Path,
    data_jsonl: Path,
    max_loops: int,
    value_prefix: str,
) -> dict[str, Any]:
    out_dir = RUN_DIR / "eval" / label
    summary = out_dir / "summary.json"
    rows = out_dir / "rows.jsonl"
    if not summary.exists():
        run(
            [
                sys.executable,
                "eval/eval_synthetic_depth_active_labels.py",
                "--model_name",
                MODEL_NAME,
                "--data_jsonl",
                path_for_cli(data_jsonl),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--output_jsonl",
                path_for_cli(rows),
                "--output_summary",
                path_for_cli(summary),
                "--loop_counts",
                ",".join(map(str, range(1, max_loops + 1))),
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
                str(RANK),
                "--lora_alpha",
                str(ALPHA),
                "--dtype",
                DTYPE,
                "--adapter_dtype",
                "float32",
                "--device",
                DEVICE,
                "--progress_every",
                "64",
            ]
        )
    return {"summary": path_for_cli(summary), "payload": read_json(summary)}


def eval_final_symbol(
    *,
    label: str,
    checkpoint: Path,
    data_jsonl: Path,
    max_loops: int,
) -> tuple[Path, Path, dict[str, Any]]:
    out_dir = RUN_DIR / "eval" / label
    summary = out_dir / "summary.json"
    rows = out_dir / "rows.jsonl"
    if not summary.exists():
        run(
            [
                sys.executable,
                "eval/eval_synthetic_depth_final_symbol.py",
                "--model_name",
                MODEL_NAME,
                "--data_jsonl",
                path_for_cli(data_jsonl),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--output_jsonl",
                path_for_cli(rows),
                "--output_summary",
                path_for_cli(summary),
                "--max_loops",
                str(max_loops),
                "--threshold",
                "0.71",
                "--prompt_style",
                "question_only",
                "--value_prefix",
                "",
                "--split",
                "6,18",
                "--bridge_projection_mode",
                "split",
                "--lora_rank",
                str(RANK),
                "--lora_alpha",
                str(ALPHA),
                "--dtype",
                DTYPE,
                "--adapter_dtype",
                "float32",
                "--device",
                DEVICE,
            ]
        )
    return rows, summary, read_json(summary)


def train_stage(
    *,
    stage: dict[str, Any],
    train_jsonl: Path,
    resume_from: Path,
    canary_path: Path,
    canary_baseline: float,
    smoke_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    name = str(stage["name"])
    output_dir = RUN_DIR / "train" / name
    expected = output_dir / f"unfrozen_recurrent_step_{int(stage['max_steps'])}.pt"
    summary_path = output_dir / "train_unfrozen_recurrent_summary.json"
    if not expected.exists():
        restore_checkpoint_from_drive(name, expected)
    if not expected.exists() or not summary_path.exists():
        config = write_train_config(
            stage=stage,
            train_jsonl=train_jsonl,
            output_dir=output_dir,
            resume_from=resume_from,
            canary_path=canary_path,
            canary_baseline=canary_baseline,
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
                DEVICE,
            ]
        )
        (output_dir / "train.log").write_text(result.stdout, encoding="utf-8")
    if not expected.exists() or not summary_path.exists():
        raise RuntimeError(f"Stage {name} did not produce its final checkpoint and summary")

    training = read_json(summary_path)
    if training.get("status") == "hard_stopped_canary":
        raise RuntimeError(f"Stage {name} hit the Tier-1 canary hard stop")
    if training.get("pretrained_base_hash_unchanged") is not True:
        raise RuntimeError(f"Stage {name} changed pretrained base parameters")
    trainable = training.get("trainable_parameters") or {}
    if int(trainable.get("total", -1)) != EXPECTED_FORWARD_ACTIVE:
        raise RuntimeError(
            f"Arm E trainable count must be the active R16 budget "
            f"{EXPECTED_FORWARD_ACTIVE}, observed={trainable}"
        )
    dose_trace = training.get("dose_trace") or []
    if stage["supervision"] == "per_loop_labels" and not dose_trace:
        raise RuntimeError(f"Stage {name} is missing per-loop dose receipts")

    checkpoint_paths = sorted(output_dir.glob("unfrozen_recurrent_step_*.pt"), key=checkpoint_step)
    smoke_receipts: list[dict[str, Any]] = []
    if smoke_path is not None:
        for checkpoint in checkpoint_paths:
            _, summary, payload = eval_final_symbol(
                label=f"{name}_step_{checkpoint_step(checkpoint)}_smoke128",
                checkpoint=checkpoint,
                data_jsonl=smoke_path,
                max_loops=int(stage["max_loops"]),
            )
            smoke_receipts.append(
                {
                    "step": checkpoint_step(checkpoint),
                    "summary": path_for_cli(summary),
                    "same_reader_total": payload["same_reader_total"],
                    "by_depth": payload["by_depth"],
                }
            )
    backup = copy_checkpoint_to_drive(expected, name)
    receipt = {
        "name": name,
        "checkpoint": path_for_cli(expected),
        "checkpoint_sha256": sha256_file(expected),
        "drive_backup": backup,
        "training_summary": path_for_cli(summary_path),
        "base_hash_start": training.get("pretrained_base_sha256_start"),
        "base_hash_end": training.get("pretrained_base_sha256_end"),
        "pretrained_base_hash_unchanged": True,
        "trainable_parameters": trainable,
        "dose_trace": dose_trace,
        "canary_trace": training.get("canary_trace") or [],
        "smoke_receipts": smoke_receipts,
    }
    write_json(RUN_DIR / "stage_receipts" / f"{name}.json", receipt)
    publish(f"Record adapter-budget Arm E stage {name} {RUN_ID} [skip ci]")
    return expected, receipt


def write_summary(payload: dict[str, Any]) -> None:
    write_json(RUN_DIR / "summary.json", payload)
    lines = [
        f"# Adapter-Budget Arm E - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Verdict: `{payload.get('verdict')}`",
        f"- Initialization: `fresh_base_qwen_surgery`",
        f"- Trainable set: R16 LoRA plus active repaired split bridge (`{EXPECTED_FORWARD_ACTIVE:,}` parameters)",
        f"- Optimizer: `AdamW`",
        "",
        "## Stages",
        "",
        "| Stage | Steps | Checkpoint SHA | Canary | Dose receipts |",
        "|---|---:|---|---|---:|",
    ]
    for stage in payload.get("stages", []):
        canaries = stage.get("canary_trace") or []
        canary_status = canaries[-1]["status"] if canaries else "manual/final"
        lines.append(
            f"| {stage['name']} | {stage['drive_backup']['path'].split('_step_')[-1].split('.')[0]} | "
            f"`{stage['checkpoint_sha256'][:12]}` | {canary_status} | {len(stage.get('dose_trace') or [])} |"
        )
    score = payload.get("adapter_budget_depth_profile") or {}
    if score:
        lines.extend(
            [
                "",
                "## Paired Phase A Read",
                "",
                f"- Arm A: `{score['arm_a']['correct']}/{score['arm_a']['total']}`.",
                f"- Arm E: `{score['arm_e']['correct']}/{score['arm_e']['total']}`.",
                f"- Verdict: `{score['verdict']}`.",
                f"- Deficit shape: `{score['deficit_shape']}`.",
            ]
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_NAME != "Qwen/Qwen2.5-0.5B-Instruct":
        raise RuntimeError(f"Arm E model is locked; observed MODEL_NAME={MODEL_NAME!r}")
    spec = locked_spec()
    write_json(RUN_DIR / "preregistration.json", spec)
    data = generate_locked_data()
    write_json(RUN_DIR / "data_manifest.json", data)

    identity_checkpoint, identity = run_identity()
    canary_path = ROOT / data["files"]["canary"]
    canary_baseline_eval = eval_active(
        label="step0_canary",
        checkpoint=identity_checkpoint,
        data_jsonl=canary_path,
        max_loops=1,
        value_prefix="name:",
    )
    canary_gate = canary_baseline_gate(canary_baseline_eval["payload"])
    if not canary_gate["passed"]:
        raise RuntimeError(f"Nonvacuous Tier-1 canary baseline failed: {canary_gate}")

    stage_receipts: list[dict[str, Any]] = []
    resume_from = identity_checkpoint
    for stage in spec["stages"]:
        train_jsonl = ROOT / data["files"][stage["name"]]
        smoke_path = (
            ROOT / data["smoke_paths"][str(stage["max_loops"])]
            if int(stage["max_loops"]) > 1
            else None
        )
        resume_from, receipt = train_stage(
            stage=stage,
            train_jsonl=train_jsonl,
            resume_from=resume_from,
            canary_path=canary_path,
            canary_baseline=float(canary_gate["accuracy"]),
            smoke_path=smoke_path,
        )
        stage_receipts.append(receipt)

    arm_e_rows_path, final_eval_summary, final_eval = eval_final_symbol(
        label="final_phase_a_1792",
        checkpoint=resume_from,
        data_jsonl=FROZEN_EVAL,
        max_loops=14,
    )
    arm_a_rows = read_jsonl(ARM_A_ROWS)
    arm_e_rows = read_jsonl(arm_e_rows_path)
    score = score_adapter_budget_arm(arm_a_rows, arm_e_rows)
    write_json(RUN_DIR / "adapter_budget_depth_profile.json", score)

    status = (
        "blocked_catastrophic_recipe_alarm"
        if score["verdict"] == "catastrophic_training_recipe_alarm"
        else "finished"
    )
    payload = {
        "kind": "stage5_adapter_budget_arm_e",
        "run_id": RUN_ID,
        "status": status,
        "verdict": score["verdict"],
        "locked_spec": spec,
        "data": data,
        "identity": identity,
        "tier1_canary_baseline": canary_gate,
        "stages": stage_receipts,
        "final_checkpoint": path_for_cli(resume_from),
        "final_checkpoint_sha256": sha256_file(resume_from),
        "final_eval_summary": path_for_cli(final_eval_summary),
        "final_eval": final_eval,
        "arm_a_rows": path_for_cli(ARM_A_ROWS),
        "arm_e_rows": path_for_cli(arm_e_rows_path),
        "adapter_budget_depth_profile": score,
        "pretrained_base_hash_unchanged": all(
            stage["pretrained_base_hash_unchanged"] for stage in stage_receipts
        ),
        "paper_claim_policy": {
            "budget_independent_allowed": score["verdict"] == "parity",
            "capacity_limited_allowed": score["verdict"] == "deficit",
            "neither_claim_allowed_on_recipe_alarm": score["verdict"]
            == "catastrophic_training_recipe_alarm",
        },
    }
    write_summary(payload)
    publish(f"Finish adapter-budget Arm E {RUN_ID} [skip ci]")
    return 2 if status.startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
