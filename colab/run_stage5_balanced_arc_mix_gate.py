"""Run an ARC-train mixed Phase 1 recovery gate from the balanced checkpoint.

This gate is for the failure mode observed after deterministic recovery:
ARC-Challenge can slightly beat base while ARC-Easy trails. Instead of more
blind Opus continuation, this runner mixes Opus reasoning traces with ARC
train-split MCQ label supervision in the exact prompt format used by eval_mcq.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_ARC_MIX_RUN_ID") or time.strftime(
    "stage5_balanced_arc_mix_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID

SOURCE_SUMMARY = Path(
    os.environ.get(
        "STAGE5_ARC_MIX_SOURCE_SUMMARY",
        "outputs/stage5/stage5_balanced_mcq_current/summary.json",
    )
)
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OPUS_DATASET_ID = os.environ.get("OPUS_DATASET_ID", "lordx64/reasoning-distill-opus-4-7-max-sft")
OPUS_LIMIT = os.environ.get("STAGE5_ARC_MIX_OPUS_LIMIT", "4000")
OPUS_VAL_FRACTION = os.environ.get("STAGE5_ARC_MIX_OPUS_VAL_FRACTION", "0.05")
MAX_TOTAL_TOKENS = os.environ.get("STAGE5_ARC_MIX_MAX_TOTAL_TOKENS", "1024")
ARC_TRAIN_LIMIT = os.environ.get("STAGE5_ARC_MIX_ARC_TRAIN_LIMIT", "0")
ARC_REPEAT = int(os.environ.get("STAGE5_ARC_MIX_ARC_REPEAT", "2"))
ARC_CHALLENGE_REPEAT = int(os.environ.get("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", str(ARC_REPEAT)))
ARC_EASY_REPEAT = int(os.environ.get("STAGE5_ARC_MIX_ARC_EASY_REPEAT", str(ARC_REPEAT)))
ARC_CHALLENGE_TARGET_LOOP = os.environ.get("STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP", "")
ARC_EASY_TARGET_LOOP = os.environ.get("STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP", "")
ARC_CHALLENGE_ROUTING_TYPE = os.environ.get("STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE", "")
ARC_EASY_ROUTING_TYPE = os.environ.get("STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE", "")
MIX_SEED = int(os.environ.get("STAGE5_ARC_MIX_SEED", "17"))
ARC_EVAL_LIMIT = int(os.environ.get("STAGE5_ARC_MIX_ARC_EVAL_LIMIT", "128"))
ARC_EVAL_CONFIG = os.environ.get("STAGE5_ARC_MIX_EVAL_CONFIG", "ARC-Challenge")
MIN_PROXY_MARGIN_DELTA = float(os.environ.get("STAGE5_ARC_MIX_MIN_MARGIN_DELTA", "-0.05"))
MAX_PROXY_PREDICTION_SHIFT = int(os.environ.get("STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT", "16"))
INCLUDE_LOOP_DIAGNOSTICS = os.environ.get("STAGE5_ARC_MIX_INCLUDE_LOOP_DIAGNOSTICS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PUSH_RESULTS = os.environ.get("STAGE5_ARC_MIX_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
SAFE_OUTPUT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
BINARY_OUTPUT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
MAX_COMMIT_ARTIFACT_BYTES = int(os.environ.get("STAGE5_ARC_MIX_COMMIT_MAX_ARTIFACT_BYTES", "25000000"))

DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")

OPUS_TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_opus_train.jsonl"
OPUS_VAL_JSONL = ROOT / "data" / f"{RUN_ID}_opus_val.jsonl"
ARC_CHALLENGE_TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_arc_challenge_train_sft.jsonl"
ARC_EASY_TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_arc_easy_train_sft.jsonl"
MIXED_TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_mixed_train.jsonl"
ARC_EVAL_SLUG = ARC_EVAL_CONFIG.lower().replace("-", "_").replace(" ", "_")
ARC_EVAL_JSONL = ROOT / "data" / f"{RUN_ID}_{ARC_EVAL_SLUG}{ARC_EVAL_LIMIT}.jsonl"


@dataclass(frozen=True)
class ArmConfig:
    name: str
    learning_rate: str
    beta: str
    steps: str
    save_every: str
    distill_enabled: str
    distill_weight: str
    distill_temperature: str
    distill_on: str


ARM_PRESETS: dict[str, ArmConfig] = {
    "arc_mix_nodistill_lr3e6": ArmConfig(
        name="arc_mix_nodistill_lr3e6",
        learning_rate="3e-6",
        beta="0.12",
        steps="150",
        save_every="50",
        distill_enabled="0",
        distill_weight="0.0",
        distill_temperature="2.0",
        distill_on="response",
    ),
    "arc_mix_response_w005_lr3e6": ArmConfig(
        name="arc_mix_response_w005_lr3e6",
        learning_rate="3e-6",
        beta="0.12",
        steps="150",
        save_every="50",
        distill_enabled="1",
        distill_weight="0.05",
        distill_temperature="2.0",
        distill_on="response",
    ),
    "arc_mix_response_w005_lr2e6": ArmConfig(
        name="arc_mix_response_w005_lr2e6",
        learning_rate="2e-6",
        beta="0.12",
        steps="150",
        save_every="50",
        distill_enabled="1",
        distill_weight="0.05",
        distill_temperature="2.0",
        distill_on="response",
    ),
    "arc_mix_response_w01_lr2e6": ArmConfig(
        name="arc_mix_response_w01_lr2e6",
        learning_rate="2e-6",
        beta="0.12",
        steps="150",
        save_every="50",
        distill_enabled="1",
        distill_weight="0.10",
        distill_temperature="2.0",
        distill_on="response",
    ),
    "arc_mix_response_w02_lr2e6": ArmConfig(
        name="arc_mix_response_w02_lr2e6",
        learning_rate="2e-6",
        beta="0.12",
        steps="150",
        save_every="50",
        distill_enabled="1",
        distill_weight="0.20",
        distill_temperature="2.0",
        distill_on="response",
    ),
}
ARM_NAMES = [
    item.strip()
    for item in os.environ.get("STAGE5_ARC_MIX_ARMS", "arc_mix_nodistill_lr3e6").split(",")
    if item.strip()
]


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_mcq(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aggregate.setdefault(str(row["aggregate"]), []).append(row)
    return {
        aggregate: {
            "correct": sum(1 for row in aggregate_rows if row["hit"]),
            "total": len(aggregate_rows),
            "accuracy": sum(1 for row in aggregate_rows if row["hit"]) / max(len(aggregate_rows), 1),
        }
        for aggregate, aggregate_rows in sorted(by_aggregate.items())
    }


def mean_accuracy(summary: dict[str, Any]) -> float | None:
    metric = summary.get("mean")
    return None if metric is None else float(metric["accuracy"])


def score_margin(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    other_scores = [float(score) for label, score in scores.items() if label != answer]
    if not other_scores:
        return None
    return float(scores[answer]) - max(other_scores)


def correct_score(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    return float(scores[answer])


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def paired_mcq_diagnostics(variant_path: Path, reference_path: Path) -> dict[str, Any]:
    reference = {row["id"]: row for row in read_jsonl(reference_path) if row["aggregate"] == "mean"}
    helped = hurt = tied = changed = 0
    margin_deltas: list[float | None] = []
    correct_score_deltas: list[float | None] = []
    reference_predictions: Counter[str] = Counter()
    variant_predictions: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    for row in read_jsonl(variant_path):
        if row["aggregate"] != "mean":
            continue
        base = reference[row["id"]]
        base_prediction = str(base.get("prediction"))
        variant_prediction = str(row.get("prediction"))
        answer = str(row.get("answer"))
        reference_predictions[base_prediction] += 1
        variant_predictions[variant_prediction] += 1
        answer_counts[answer] += 1
        if variant_prediction != base_prediction:
            changed += 1
        if row["hit"] and not base["hit"]:
            helped += 1
        elif base["hit"] and not row["hit"]:
            hurt += 1
        else:
            tied += 1

        base_margin = score_margin(base)
        row_margin = score_margin(row)
        margin_deltas.append(None if base_margin is None or row_margin is None else row_margin - base_margin)
        base_correct_score = correct_score(base)
        row_correct_score = correct_score(row)
        correct_score_deltas.append(
            None
            if base_correct_score is None or row_correct_score is None
            else row_correct_score - base_correct_score
        )

    labels = sorted(set(reference_predictions) | set(variant_predictions) | set(answer_counts))
    prediction_count_delta = {
        label: int(variant_predictions[label] - reference_predictions[label]) for label in labels
    }
    max_abs_prediction_count_delta = max(
        [abs(value) for value in prediction_count_delta.values()] or [0]
    )
    mean_margin_delta = finite_mean(margin_deltas)
    calibration_ok = (
        mean_margin_delta is None or mean_margin_delta >= MIN_PROXY_MARGIN_DELTA
    ) and max_abs_prediction_count_delta <= MAX_PROXY_PREDICTION_SHIFT
    return {
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "prediction_changes": changed,
        "mean_margin_delta": mean_margin_delta,
        "mean_correct_score_delta": finite_mean(correct_score_deltas),
        "reference_prediction_counts": dict(reference_predictions),
        "variant_prediction_counts": dict(variant_predictions),
        "answer_counts": dict(answer_counts),
        "prediction_count_delta": prediction_count_delta,
        "max_abs_prediction_count_delta": max_abs_prediction_count_delta,
        "calibration_ok": calibration_ok,
        "calibration_thresholds": {
            "min_mean_margin_delta": MIN_PROXY_MARGIN_DELTA,
            "max_abs_prediction_count_delta": MAX_PROXY_PREDICTION_SHIFT,
        },
    }


def helped_hurt(variant_path: Path, reference_path: Path) -> dict[str, Any]:
    return paired_mcq_diagnostics(variant_path, reference_path)


def selected_checkpoint(source_payload: dict[str, Any]) -> Path:
    best = source_payload.get("best_checkpoint") or {}
    checkpoint = best.get("checkpoint") or source_payload.get("checkpoint") or source_payload.get("selected_checkpoint")
    if not checkpoint and source_payload.get("source_summary"):
        source = read_json(resolve_path(str(source_payload["source_summary"])))
        checkpoint = source.get("checkpoint")
    if not checkpoint:
        raise ValueError("Source summary does not contain best_checkpoint.checkpoint or checkpoint")
    return resolve_path(str(checkpoint))


def checkpoint_run_id(checkpoint: str | Path) -> str:
    parts = list(Path(checkpoint).parts)
    for idx, part in enumerate(parts):
        if part == "stage5" and idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(f"Could not infer Stage 5 run id from checkpoint path: {checkpoint}")


def arm_config(name: str) -> ArmConfig:
    try:
        return ARM_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ARC-mix arm {name!r}; choose one of {sorted(ARM_PRESETS)}") from exc


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
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
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if log_name:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def prepare_opus() -> None:
    if OPUS_TRAIN_JSONL.exists() and OPUS_VAL_JSONL.exists():
        return
    run(
        [
            sys.executable,
            "training/prepare_hf_reasoning_jsonl.py",
            "--dataset_id",
            OPUS_DATASET_ID,
            "--tokenizer_name",
            MODEL_NAME,
            "--output_jsonl",
            path_for_cli(OPUS_TRAIN_JSONL),
            "--val_jsonl",
            path_for_cli(OPUS_VAL_JSONL),
            "--limit",
            OPUS_LIMIT,
            "--val_fraction",
            OPUS_VAL_FRACTION,
            "--max_total_tokens",
            MAX_TOTAL_TOKENS,
        ],
        log_name="prepare_opus.log",
    )


def prepare_arc_sft(
    config: str,
    output: Path,
    *,
    target_loop: str = "",
    routing_type: str = "",
) -> None:
    if output.exists():
        return
    cmd = [
        sys.executable,
        "training/prepare_arc_mcq_sft_jsonl.py",
        "--config",
        config,
        "--split",
        "train",
        "--tokenizer_name",
        MODEL_NAME,
        "--output_jsonl",
        path_for_cli(output),
        "--seed",
        str(MIX_SEED),
        "--prompt_style",
        "with_options",
        "--score_target",
        "label",
        "--max_total_tokens",
        "512",
    ]
    if ARC_TRAIN_LIMIT.strip().lower() not in {"", "0", "none", "all", "full"}:
        cmd.extend(["--limit", ARC_TRAIN_LIMIT])
    if target_loop.strip():
        cmd.extend(["--target_loop_count", target_loop.strip()])
    if routing_type.strip():
        cmd.extend(["--routing_type", routing_type.strip()])
    run(cmd, log_name=f"prepare_{output.stem}.log")


def prepare_arc_eval(path: Path, *, limit: int | None) -> None:
    cmd = [
        sys.executable,
        "eval/prepare_arc_mcq.py",
        "--config",
        ARC_EVAL_CONFIG,
        "--split",
        "validation",
        "--seed",
        str(MIX_SEED),
        "--output_jsonl",
        path_for_cli(path),
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    run(cmd, log_name=f"prepare_{path.stem}.log")


def eval_arc(label: str, mode: str, data_jsonl: Path, checkpoint: Path | None = None) -> Path:
    output = RUN_DIR / f"{data_jsonl.stem}_{label}.jsonl"
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--data_jsonl",
        path_for_cli(data_jsonl),
        "--prompt_style",
        "with_options",
        "--score_target",
        "label",
        "--mode",
        mode,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--seed",
        "0",
        "--aggregate",
        "mean",
    ]
    if mode == "phase1":
        assert checkpoint is not None
        cmd += [
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_trajectories",
            "1",
        ]
        if INCLUDE_LOOP_DIAGNOSTICS:
            cmd.append("--include_loop_diagnostics")
    run(cmd + ["--output_jsonl", path_for_cli(output)], log_name=f"{data_jsonl.stem}_{label}.log")
    return output


def build_mixed_train() -> dict[str, Any]:
    prepare_opus()
    prepare_arc_sft(
        "ARC-Challenge",
        ARC_CHALLENGE_TRAIN_JSONL,
        target_loop=ARC_CHALLENGE_TARGET_LOOP,
        routing_type=ARC_CHALLENGE_ROUTING_TYPE,
    )
    prepare_arc_sft(
        "ARC-Easy",
        ARC_EASY_TRAIN_JSONL,
        target_loop=ARC_EASY_TARGET_LOOP,
        routing_type=ARC_EASY_ROUTING_TYPE,
    )

    opus_rows = read_jsonl(OPUS_TRAIN_JSONL)
    arc_challenge_rows = read_jsonl(ARC_CHALLENGE_TRAIN_JSONL)
    arc_easy_rows = read_jsonl(ARC_EASY_TRAIN_JSONL)
    mixed = [*opus_rows]
    for _ in range(max(1, ARC_CHALLENGE_REPEAT)):
        mixed.extend(arc_challenge_rows)
    for _ in range(max(1, ARC_EASY_REPEAT)):
        mixed.extend(arc_easy_rows)
    random.Random(MIX_SEED).shuffle(mixed)
    write_jsonl(MIXED_TRAIN_JSONL, mixed)
    return {
        "opus_rows": len(opus_rows),
        "arc_challenge_rows": len(arc_challenge_rows),
        "arc_easy_rows": len(arc_easy_rows),
        "arc_repeat": ARC_REPEAT,
        "arc_challenge_repeat": ARC_CHALLENGE_REPEAT,
        "arc_easy_repeat": ARC_EASY_REPEAT,
        "arc_challenge_target_loop": ARC_CHALLENGE_TARGET_LOOP or None,
        "arc_easy_target_loop": ARC_EASY_TARGET_LOOP or None,
        "arc_challenge_routing_type": ARC_CHALLENGE_ROUTING_TYPE or None,
        "arc_easy_routing_type": ARC_EASY_ROUTING_TYPE or None,
        "mixed_rows": len(mixed),
        "mixed_train_jsonl": path_for_cli(MIXED_TRAIN_JSONL),
        "opus_val_jsonl": path_for_cli(OPUS_VAL_JSONL),
        "arc_prompt_style": "with_options",
        "arc_score_target": "label",
        "arc_eval_config": ARC_EVAL_CONFIG,
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def train_arm(config: ArmConfig, *, resume_checkpoint: Path) -> list[Path]:
    output_dir = RUN_DIR / config.name / "phase1"
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": 512,
        "max_loops": 4,
        "initial_halt_prob": 0.15,
        "beta": float(config.beta),
        "batch_size": 1,
        "learning_rate": float(config.learning_rate),
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": int(config.steps),
        "save_every": int(config.save_every),
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(resume_checkpoint),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
        "distillation": {
            "enabled": config.distill_enabled == "1",
            "weight": float(config.distill_weight),
            "temperature": float(config.distill_temperature),
            "on": config.distill_on,
            "teacher_model_name": MODEL_NAME,
            "dtype": DTYPE,
        },
    }
    cfg_path = RUN_DIR / config.name / "phase1_continue.yaml"
    write_yaml(cfg_path, cfg)
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(MIXED_TRAIN_JSONL),
            "--device",
            DEVICE,
        ],
        log_name=f"{config.name}_train.log",
    )
    checkpoints = sorted(output_dir.glob("phase1_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints produced under {output_dir}")
    return checkpoints


def eval_jsonl_with_val(label: str, checkpoint: Path, *, beta: str) -> dict[str, float]:
    proc = run(
        [
            sys.executable,
            "eval/eval_jsonl.py",
            "--model_name",
            MODEL_NAME,
            "--data_jsonl",
            path_for_cli(OPUS_VAL_JSONL),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--split",
            "6,18",
            "--max_loops",
            "4",
            "--max_length",
            "512",
            "--beta",
            beta,
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name=f"{label}_val.log",
    )
    metrics: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return metrics


def evaluate_arm(config: ArmConfig, *, resume_checkpoint: Path, checkpoints: list[Path]) -> dict[str, Any]:
    base_arc_path = eval_arc(f"{config.name}_base_label", "base", ARC_EVAL_JSONL)
    start_arc_path = eval_arc(f"{config.name}_start_label", "phase1", ARC_EVAL_JSONL, resume_checkpoint)
    start_val = eval_jsonl_with_val(f"{config.name}_start", resume_checkpoint, beta=config.beta)
    checkpoint_rows = []
    for checkpoint in checkpoints:
        label = f"{config.name}_{checkpoint.stem}"
        arc_path = eval_arc(f"{label}_label", "phase1", ARC_EVAL_JSONL, checkpoint)
        checkpoint_rows.append(
            {
                "checkpoint": path_for_cli(checkpoint),
                "val": eval_jsonl_with_val(label, checkpoint, beta=config.beta),
                "arc_path": path_for_cli(arc_path),
                "arc": summarize_mcq(arc_path),
                "comparison_to_start": helped_hurt(arc_path, start_arc_path),
                "comparison_to_base": helped_hurt(arc_path, base_arc_path),
            }
        )
    best = max(checkpoint_rows, key=lambda item: mean_accuracy(item["arc"]) or -1.0)
    return {
        "arm": config.name,
        "base_arc": summarize_mcq(base_arc_path),
        "phase1_start": {
            "checkpoint": path_for_cli(resume_checkpoint),
            "val": start_val,
            "arc": summarize_mcq(start_arc_path),
        },
        "checkpoints": checkpoint_rows,
        "best_checkpoint": best,
        "distillation": {
            "enabled": config.distill_enabled == "1",
            "weight": float(config.distill_weight),
            "temperature": float(config.distill_temperature),
            "on": config.distill_on,
        },
        "learning_rate": float(config.learning_rate),
        "beta": float(config.beta),
        "steps": int(config.steps),
    }


def correct(summary: dict[str, Any]) -> int:
    return int(((summary.get("mean") or {}).get("correct", 0)) or 0)


def best_calibration(best_arm: dict[str, Any] | None) -> dict[str, Any]:
    if best_arm is None:
        return {}
    return dict((best_arm.get("best_checkpoint") or {}).get("comparison_to_base") or {})


def build_summary(
    *,
    source_summary: Path,
    source_payload: dict[str, Any],
    resume_checkpoint: Path,
    data_summary: dict[str, Any],
    arms: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(
        arms,
        key=lambda row: (
            correct(row["best_checkpoint"]["arc"]) - correct(row["phase1_start"]["arc"]),
            correct(row["best_checkpoint"]["arc"]) - correct(row["base_arc"]),
            mean_accuracy(row["best_checkpoint"]["arc"]) or 0.0,
            row["arm"],
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    if best is None:
        status = "no_arms"
        next_step = "Run at least one ARC-mix arm."
        decision = "rerun_with_valid_arm"
        blocked_reason = "No ARC-mix arms were evaluated."
    else:
        lift = correct(best["best_checkpoint"]["arc"]) - correct(best["phase1_start"]["arc"])
        gap = correct(best["best_checkpoint"]["arc"]) - correct(best["base_arc"])
        calibration = best_calibration(best)
        calibration_ok = bool(calibration.get("calibration_ok", True))
        if lift > 0 and calibration_ok:
            status = "proxy_lift"
            next_step = "Run full ARC-Easy and ARC-Challenge balanced benchmark on the best ARC-mix checkpoint."
            decision = "run_full_balanced_assessment"
            blocked_reason = None
        elif gap >= 0 and calibration_ok:
            status = "proxy_matches_base"
            next_step = "Run full balanced benchmark on the best ARC-mix checkpoint; proxy no longer trails base."
            decision = "run_full_balanced_assessment"
            blocked_reason = None
        elif lift > 0:
            status = "proxy_lift_calibration_warning"
            next_step = (
                "Do not run the full paid assessment yet; the proxy lifted accuracy but degraded "
                "base-comparison calibration. Increase preservation/distillation or inspect answer-prior drift."
            )
            decision = "stop_for_calibration_repair"
            blocked_reason = "Proxy lifted accuracy but failed the calibration-preservation threshold."
        elif gap >= 0:
            status = "proxy_matches_base_calibration_warning"
            next_step = (
                "Do not run the full paid assessment yet; the proxy matched base accuracy but degraded "
                "base-comparison calibration. Tighten the recovery objective before spending A100 on confirmation."
            )
            decision = "stop_for_calibration_repair"
            blocked_reason = "Proxy matched base accuracy but failed the calibration-preservation threshold."
        else:
            status = "no_proxy_lift"
            next_step = "Do not extend this ARC-mix setting; inspect failures or revise supervision mix."
            decision = "stop_and_revise_objective"
            blocked_reason = "Best ARC-mix checkpoint did not improve over the recurrent start or close the base gap."

    return {
        "run_id": RUN_ID,
        "kind": "stage5_balanced_arc_mix_gate",
        "objective_rationale": {
            "failure_mode": (
                "The previous proxy-selected recurrent checkpoint lost full balanced ARC points through "
                "answer-calibration drift: lower correct-answer margins and answer-prior shift."
            ),
            "proxy_hypothesis": (
                "Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only "
                "frozen-base KL distillation to preserve the base model's answer-token distribution."
            ),
            "response_distillation_reason": (
                "ARC MCQ SFT rows use label-only completions, so response-only distillation is concentrated "
                "on the option label decision rather than the prompt text."
            ),
        },
        "source_summary": path_for_cli(source_summary),
        "source_status": source_payload.get("status"),
        "resume_checkpoint": path_for_cli(resume_checkpoint),
        "resume_run_id": checkpoint_run_id(resume_checkpoint),
        "data": data_summary,
        "arc_eval_limit": ARC_EVAL_LIMIT,
        "arc_eval_config": ARC_EVAL_CONFIG,
        "proxy_calibration_thresholds": {
            "min_mean_margin_delta": MIN_PROXY_MARGIN_DELTA,
            "max_abs_prediction_count_delta": MAX_PROXY_PREDICTION_SHIFT,
        },
        "status": status,
        "passed": status in {"proxy_lift", "proxy_matches_base"},
        "decision": decision,
        "blocked_reason": blocked_reason,
        "next_step": next_step,
        "best_arm": best,
        "arms": ranked,
    }


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Balanced ARC-Mix Gate - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Blocked reason: {payload['blocked_reason'] or 'none'}",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Resume checkpoint: `{payload['resume_checkpoint']}`",
        f"- Mixed rows: `{payload['data']['mixed_rows']}`",
        f"- Proxy eval config: `{payload['arc_eval_config']}`",
        f"- Calibration thresholds: mean margin delta >= `{payload['proxy_calibration_thresholds']['min_mean_margin_delta']}`, "
        f"max prediction-count shift <= `{payload['proxy_calibration_thresholds']['max_abs_prediction_count_delta']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Objective Rationale",
        "",
        f"- Failure mode: {payload['objective_rationale']['failure_mode']}",
        f"- Proxy hypothesis: {payload['objective_rationale']['proxy_hypothesis']}",
        f"- Response distillation reason: {payload['objective_rationale']['response_distillation_reason']}",
        "",
        "## Arms",
        "",
        "| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["arms"]:
        best_arc = row["best_checkpoint"]["arc"]["mean"]
        start_arc = row["phase1_start"]["arc"]["mean"]
        base_arc = row["base_arc"]["mean"]
        calibration = row["best_checkpoint"].get("comparison_to_base") or {}
        margin = calibration.get("mean_margin_delta")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['arm']}`",
                    f"{best_arc['correct']}/{best_arc['total']}",
                    f"{start_arc['correct']}/{start_arc['total']}",
                    f"{base_arc['correct']}/{base_arc['total']}",
                    str(int(best_arc["correct"]) - int(start_arc["correct"])),
                    str(int(best_arc["correct"]) - int(base_arc["correct"])),
                    "n/a" if margin is None else f"{float(margin):.4f}",
                    str(calibration.get("max_abs_prediction_count_delta", "n/a")),
                    "`ok`" if calibration.get("calibration_ok", True) else "`warning`",
                    f"`{row['best_checkpoint']['checkpoint']}`",
                ]
            )
            + " |"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def backup_to_drive() -> None:
    try:
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except Exception as exc:
        print(f"Drive mount skipped/failed: {exc}")
        return
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for source in [OPUS_TRAIN_JSONL, OPUS_VAL_JSONL, ARC_CHALLENGE_TRAIN_JSONL, ARC_EASY_TRAIN_JSONL, MIXED_TRAIN_JSONL]:
        if source.exists():
            target = backup / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"backed_up_to={backup}")


def is_safe_output_artifact(path: Path, *, max_bytes: int = MAX_COMMIT_ARTIFACT_BYTES) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix in BINARY_OUTPUT_SUFFIXES:
        return False
    if suffix not in SAFE_OUTPUT_SUFFIXES:
        return False
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def committable_run_files() -> list[str]:
    if not RUN_DIR.exists():
        return []
    files = [
        path.relative_to(ROOT).as_posix()
        for path in RUN_DIR.rglob("*")
        if is_safe_output_artifact(path)
    ]
    return sorted(files)


def git_add_files(paths: list[str]) -> None:
    chunk_size = 100
    for index in range(0, len(paths), chunk_size):
        run(["git", "add", "-f", *paths[index : index + chunk_size]], check=False)


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    files = committable_run_files()
    if not files:
        print("No safe ARC-mix gate text artifacts to commit.")
        return
    git_add_files(files)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-mix gate outputs changed.")
        return
    run(["git", "commit", "-m", f"Record balanced Stage 5 ARC-mix gate {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_payload = read_json(SOURCE_SUMMARY)
    resume_checkpoint = selected_checkpoint(source_payload)
    restore_checkpoint_if_needed(resume_checkpoint, run_id=checkpoint_run_id(resume_checkpoint))
    data_summary = build_mixed_train()
    prepare_arc_eval(ARC_EVAL_JSONL, limit=ARC_EVAL_LIMIT)

    arms = []
    for name in ARM_NAMES:
        config = arm_config(name)
        checkpoints = train_arm(config, resume_checkpoint=resume_checkpoint)
        arms.append(evaluate_arm(config, resume_checkpoint=resume_checkpoint, checkpoints=checkpoints))

    payload = build_summary(
        source_summary=SOURCE_SUMMARY,
        source_payload=source_payload,
        resume_checkpoint=resume_checkpoint,
        data_summary=data_summary,
        arms=arms,
    )
    write_report(payload)
    backup_to_drive()
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
