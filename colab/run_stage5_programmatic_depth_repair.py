"""Run a bounded Phase 1 repair on verified programmatic direct/deep data.

This runner is intentionally narrower than the ARC-mix repair gate. It creates
constructed arithmetic-chain curriculum rows on CPU, exports only positive SFT
traces, runs one short deterministic Phase 1 continuation, and evaluates on a
held-out constructed validation split. Use ARC/benchmark gates to decide whether
the checkpoint is actually useful; this runner is a cheap calibration lever, not
a claim of benchmark progress by itself.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_PROGRAMMATIC_DEPTH_RUN_ID") or time.strftime(
    "stage5_programmatic_depth_repair_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")

NUM_DIRECT = int(os.environ.get("STAGE5_PROGRAMMATIC_NUM_DIRECT", "1200"))
NUM_DEEP_NARROW = int(os.environ.get("STAGE5_PROGRAMMATIC_NUM_DEEP_NARROW", "800"))
VAL_DIRECT = int(os.environ.get("STAGE5_PROGRAMMATIC_VAL_DIRECT", "120"))
VAL_DEEP_NARROW = int(os.environ.get("STAGE5_PROGRAMMATIC_VAL_DEEP_NARROW", "120"))
SEED = int(os.environ.get("STAGE5_PROGRAMMATIC_SEED", "101"))
VAL_SEED = int(os.environ.get("STAGE5_PROGRAMMATIC_VAL_SEED", "202"))

MAX_STEPS = int(os.environ.get("STAGE5_PROGRAMMATIC_MAX_STEPS", "100"))
SAVE_EVERY = int(os.environ.get("STAGE5_PROGRAMMATIC_SAVE_EVERY", "50"))
LEARNING_RATE = float(os.environ.get("STAGE5_PROGRAMMATIC_LR", "2e-6"))
BETA = float(os.environ.get("STAGE5_PROGRAMMATIC_BETA", "0.12"))
DISTILL_WEIGHT = float(os.environ.get("STAGE5_PROGRAMMATIC_DISTILL_WEIGHT", "0.20"))
DISTILL_TEMPERATURE = float(os.environ.get("STAGE5_PROGRAMMATIC_DISTILL_TEMPERATURE", "2.0"))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
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
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def resolve_resume_checkpoint() -> Path:
    explicit = os.environ.get("STAGE5_PROGRAMMATIC_RESUME_CHECKPOINT", "").strip()
    if explicit:
        return ROOT / explicit if not Path(explicit).is_absolute() else Path(explicit)

    source_summary = os.environ.get("STAGE5_PROGRAMMATIC_SOURCE_SUMMARY", "").strip()
    if not source_summary:
        raise ValueError(
            "Set STAGE5_PROGRAMMATIC_RESUME_CHECKPOINT or "
            "STAGE5_PROGRAMMATIC_SOURCE_SUMMARY before running this repair."
        )
    source_path = ROOT / source_summary if not Path(source_summary).is_absolute() else Path(source_summary)
    payload = read_json(source_path)
    checkpoint = checkpoint_from_payload(payload)
    if not checkpoint:
        raise ValueError(f"Source summary does not contain a checkpoint: {source_path}")
    return ROOT / checkpoint if not Path(str(checkpoint)).is_absolute() else Path(str(checkpoint))


def checkpoint_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("best_checkpoint", "checkpoint", "selected_checkpoint"):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("checkpoint"):
            return str(value["checkpoint"])
        if isinstance(value, str) and value:
            return value

    best_arm = payload.get("best_arm")
    if isinstance(best_arm, dict):
        nested = best_arm.get("best_checkpoint")
        if isinstance(nested, dict) and nested.get("checkpoint"):
            return str(nested["checkpoint"])

    arc_mix = payload.get("arc_mix")
    if isinstance(arc_mix, dict):
        checkpoint = checkpoint_from_payload(arc_mix)
        if checkpoint:
            return checkpoint
    return None


def checkpoint_run_id(checkpoint: Path) -> str:
    parts = list(checkpoint.parts)
    for index, part in enumerate(parts):
        if part == "stage5" and index + 1 < len(parts):
            return parts[index + 1]
    return os.environ.get("STAGE5_PROGRAMMATIC_RESUME_RUN_ID", RUN_ID)


def generate_split(*, prefix: str, num_direct: int, num_deep_narrow: int, seed: int) -> tuple[Path, Path, Path]:
    typed = RUN_DIR / f"{prefix}_typed.jsonl"
    typed_report = RUN_DIR / f"{prefix}_typed_report.json"
    sft = RUN_DIR / f"{prefix}_sft.jsonl"
    sft_report = RUN_DIR / f"{prefix}_sft_report.json"
    run(
        [
            sys.executable,
            "training/generate_programmatic_curriculum.py",
            "--output_jsonl",
            path_for_cli(typed),
            "--report_json",
            path_for_cli(typed_report),
            "--num_direct",
            str(num_direct),
            "--num_deep_narrow",
            str(num_deep_narrow),
            "--seed",
            str(seed),
        ],
        log_name=f"generate_{prefix}.log",
    )
    run(
        [
            sys.executable,
            "training/prepare_curriculum_jsonl.py",
            "--input_jsonl",
            path_for_cli(typed),
            "--output_jsonl",
            path_for_cli(sft),
            "--report_json",
            path_for_cli(sft_report),
            "--modes",
            "direct,deep_narrow",
        ],
        log_name=f"convert_{prefix}.log",
    )
    return typed, sft, sft_report


def write_training_config(resume_checkpoint: Path, train_jsonl: Path) -> Path:
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": 512,
        "max_loops": 4,
        "initial_halt_prob": 0.15,
        "beta": BETA,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": MAX_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(RUN_DIR / "phase1"),
        "resume_from": path_for_cli(resume_checkpoint),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
        "distillation": {
            "enabled": DISTILL_WEIGHT > 0,
            "weight": DISTILL_WEIGHT,
            "temperature": DISTILL_TEMPERATURE,
            "on": "response",
            "teacher_model_name": MODEL_NAME,
            "dtype": DTYPE,
        },
    }
    path = RUN_DIR / "phase1_programmatic_depth.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def eval_jsonl(label: str, checkpoint: Path, val_jsonl: Path) -> dict[str, float]:
    proc = run(
        [
            sys.executable,
            "eval/eval_jsonl.py",
            "--model_name",
            MODEL_NAME,
            "--data_jsonl",
            path_for_cli(val_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--split",
            "6,18",
            "--max_loops",
            "4",
            "--max_length",
            "512",
            "--beta",
            str(BETA),
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name=f"{label}_eval.log",
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


def produced_checkpoints() -> list[Path]:
    out = RUN_DIR / "phase1"
    return sorted(out.glob("phase1_step_*.pt"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def write_report(payload: dict[str, Any]) -> None:
    write_json(RUN_DIR / "summary.json", payload)
    lines = [
        f"# Stage 5 Programmatic Depth Repair - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Resume checkpoint: `{payload['resume_checkpoint']}`",
        f"- Train SFT rows: `{payload['train_report']['exported_examples']}`",
        f"- Val SFT rows: `{payload['val_report']['exported_examples']}`",
        f"- Best checkpoint: `{payload.get('best_checkpoint')}`",
        f"- Start mean loops: `{payload['start_eval'].get('mean_expected_loops')}`",
        f"- Best mean loops: `{payload['best_eval'].get('mean_expected_loops')}`",
        f"- Next step: {payload['next_step']}",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    resume_checkpoint = resolve_resume_checkpoint()
    restore_checkpoint_if_needed(resume_checkpoint, run_id=checkpoint_run_id(resume_checkpoint))

    _train_typed, train_sft, train_report_path = generate_split(
        prefix="train",
        num_direct=NUM_DIRECT,
        num_deep_narrow=NUM_DEEP_NARROW,
        seed=SEED,
    )
    _val_typed, val_sft, val_report_path = generate_split(
        prefix="val",
        num_direct=VAL_DIRECT,
        num_deep_narrow=VAL_DEEP_NARROW,
        seed=VAL_SEED,
    )

    start_eval = eval_jsonl("start", resume_checkpoint, val_sft)
    cfg_path = write_training_config(resume_checkpoint, train_sft)
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(train_sft),
            "--device",
            DEVICE,
        ],
        log_name="train_phase1.log",
    )
    checkpoints = produced_checkpoints()
    if not checkpoints:
        raise FileNotFoundError(RUN_DIR / "phase1")
    evals = [{"checkpoint": path_for_cli(path), "metrics": eval_jsonl(path.stem, path, val_sft)} for path in checkpoints]
    best = min(evals, key=lambda row: row["metrics"].get("loss", float("inf")))
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_programmatic_depth_repair",
        "status": "complete",
        "resume_checkpoint": path_for_cli(resume_checkpoint),
        "train_sft": path_for_cli(train_sft),
        "val_sft": path_for_cli(val_sft),
        "train_report": read_json(train_report_path),
        "val_report": read_json(val_report_path),
        "config": path_for_cli(cfg_path),
        "start_eval": start_eval,
        "checkpoint_evals": evals,
        "best_checkpoint": best["checkpoint"],
        "best_eval": best["metrics"],
        "next_step": (
            "Use this only as a calibration ingredient. Confirm any selected checkpoint with "
            "the ARC routing/benchmark gate before proceeding to particles or wider data."
        ),
    }
    write_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
