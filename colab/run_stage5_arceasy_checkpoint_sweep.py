"""Sweep deterministic recovered Phase 1 checkpoints on ARC-Easy.

The ARC-Challenge-selected checkpoint beat base on full ARC-Challenge but
regressed on ARC-Easy. This runner evaluates the recovered parent plus the
extension ladder checkpoints against one shared ARC-Easy base run, so we can
tell whether the easy-question regression came from the surgery itself or from
checkpoint selection during continued training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    DEFAULT_CHECKPOINT_REL,
    DEFAULT_RECOVERED_RUN_ID,
    path_for_cli,
    restore_checkpoint_if_needed,
)
from colab.run_stage5_benchmark_suite import paired_arm_summaries, summarize_rows  # noqa: E402


RUN_ID = os.environ.get("STAGE5_ARCEASY_SWEEP_RUN_ID") or time.strftime(
    "stage5_arceasy_sweep_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
DATA_JSONL = ROOT / "data" / "stage5_benchmark_suite" / RUN_ID / "arc_easy.jsonl"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
PARENT_RUN_ID = os.environ.get("STAGE5_SWEEP_PARENT_RUN_ID", DEFAULT_RECOVERED_RUN_ID)
PARENT_CHECKPOINT = Path(os.environ.get("STAGE5_SWEEP_PARENT_CHECKPOINT", DEFAULT_CHECKPOINT_REL))
EXTEND_RUN_ID = os.environ.get(
    "STAGE5_SWEEP_EXTEND_RUN_ID",
    "stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143",
)
EXTEND_STEPS = os.environ.get("STAGE5_SWEEP_EXTEND_STEPS", "50,100,150,200,250")
ARC_EASY_LIMIT = os.environ.get("STAGE5_SWEEP_ARC_EASY_LIMIT", "full")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_SWEEP_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_steps(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_arms() -> list[tuple[str, str, Path]]:
    parent = resolve(PARENT_CHECKPOINT)
    arms: list[tuple[str, str, Path]] = [("parent", PARENT_RUN_ID, parent)]
    for step in parse_steps(EXTEND_STEPS):
        checkpoint = ROOT / "outputs" / "stage5" / EXTEND_RUN_ID / "phase1" / f"phase1_step_{step}.pt"
        arms.append((f"step_{step:03d}", EXTEND_RUN_ID, checkpoint))
    return arms


def restore_checkpoints() -> list[tuple[str, Path]]:
    restored: list[tuple[str, Path]] = []
    for label, run_id, checkpoint in checkpoint_arms():
        restore_checkpoint_if_needed(checkpoint, run_id=run_id)
        restored.append((label, checkpoint))
    return restored


def prepare_arc_easy() -> None:
    cmd = [
        sys.executable,
        "eval/prepare_arc_mcq.py",
        "--config",
        "ARC-Easy",
        "--split",
        "validation",
        "--seed",
        "0",
        "--output_jsonl",
        path_for_cli(DATA_JSONL),
    ]
    if ARC_EASY_LIMIT.strip().lower() not in {"", "full", "all", "none", "0"}:
        cmd.extend(["--limit", ARC_EASY_LIMIT])
    run(cmd, log_name="prepare_arc_easy.log")


def eval_base() -> Path:
    output = RUN_DIR / "base.jsonl"
    if output.exists():
        output.unlink()
    run(
        [
            sys.executable,
            "eval/eval_mcq.py",
            "--data_jsonl",
            path_for_cli(DATA_JSONL),
            "--prompt_style",
            "with_options",
            "--score_target",
            "label",
            "--aggregate",
            "mean",
            "--mode",
            "base",
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--seed",
            "0",
            "--output_jsonl",
            path_for_cli(output),
        ],
        log_name="base.log",
    )
    return output


def eval_phase1(label: str, checkpoint: Path) -> Path:
    output = RUN_DIR / f"{label}.jsonl"
    if output.exists():
        output.unlink()
    run(
        [
            sys.executable,
            "eval/eval_mcq.py",
            "--data_jsonl",
            path_for_cli(DATA_JSONL),
            "--prompt_style",
            "with_options",
            "--score_target",
            "label",
            "--aggregate",
            "mean",
            "--mode",
            "phase1",
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_trajectories",
            "1",
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--seed",
            "0",
            "--output_jsonl",
            path_for_cli(output),
        ],
        log_name=f"{label}.log",
    )
    return output


def build_summary(base_path: Path, arm_outputs: list[tuple[str, Path, Path]]) -> dict[str, Any]:
    base_rows = read_jsonl(base_path)
    base_summary = summarize_rows(base_rows).get("mean", {})
    arms: list[dict[str, Any]] = []
    for label, checkpoint, output in arm_outputs:
        rows = read_jsonl(output)
        row_summary = summarize_rows(rows).get("mean", {})
        paired = paired_arm_summaries(base_rows, rows).get("mean", {})
        arms.append(
            {
                "label": label,
                "checkpoint": path_for_cli(checkpoint),
                "output": path_for_cli(output),
                "summary": row_summary,
                "delta_vs_base": int(row_summary.get("correct", 0)) - int(base_summary.get("correct", 0)),
                "paired_vs_base": paired,
            }
        )
    return {
        "run_id": RUN_ID,
        "kind": "stage5_arceasy_checkpoint_sweep",
        "arc_easy_limit": ARC_EASY_LIMIT,
        "base": {"output": path_for_cli(base_path), "summary": base_summary},
        "arms": arms,
    }


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# ARC-Easy Checkpoint Sweep - {RUN_ID}",
        "",
        f"- ARC-Easy limit: `{payload['arc_easy_limit']}`",
        f"- Base: `{payload['base']['summary'].get('correct')}/{payload['base']['summary'].get('total')}`",
        "",
        "| arm | correct | delta vs base | W/L/T | p | checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in payload["arms"]:
        summary = arm["summary"]
        paired = arm["paired_vs_base"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{arm['label']}`",
                    f"{summary.get('correct')}/{summary.get('total')}",
                    str(arm["delta_vs_base"]),
                    f"{paired.get('wins')}/{paired.get('losses')}/{paired.get('ties')}",
                    str(paired.get("sign_test_p_value")),
                    f"`{arm['checkpoint']}`",
                ]
            )
            + " |"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No ARC-Easy sweep outputs changed.")
        return
    run(["git", "commit", "-m", f"Record ARC-Easy checkpoint sweep {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    checkpoints = restore_checkpoints()
    prepare_arc_easy()
    base_path = eval_base()
    arm_outputs = [(label, checkpoint, eval_phase1(label, checkpoint)) for label, checkpoint in checkpoints]
    payload = build_summary(base_path, arm_outputs)
    write_report(payload)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
