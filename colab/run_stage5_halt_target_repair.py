"""Run a no-Drive MCQ ladder repair pass with explicit halting supervision.

This launcher is for the small, cheap follow-up after the MCQ ladder SFT run:
the model learned the answer traces, but direct and deep rows both halted around
the same loop depth. It reconstructs the tracked MCQ ladder curriculum from
``scored_capability_rows.jsonl`` and resumes Phase 1 SFT with
``halt_target_nll_weight`` enabled.
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


RUN_ID = os.environ.get("STAGE5_HALT_TARGET_REPAIR_RUN_ID") or time.strftime(
    "stage5_halt_target_repair_%Y%m%d_%H%M%S"
)
SOURCE_SUMMARY = os.environ.get(
    "STAGE5_HALT_TARGET_REPAIR_SOURCE_SUMMARY",
    "outputs/stage5/stage5_mcq_ladder_sft_nodrive_20260623_052428/summary.json",
)
SCORED_ROWS = os.environ.get(
    "STAGE5_HALT_TARGET_REPAIR_SCORED_ROWS",
    "data/stage5_capability_ladder/stage5_capability_ladder_mcq_probe_20260623_023702/scored_capability_rows.jsonl",
)
WORK_DIR = os.environ.get(
    "STAGE5_HALT_TARGET_REPAIR_WORK_DIR",
    f"data/curriculum/{RUN_ID}_mcq_ladder",
)
MODEL_LADDER = os.environ.get(
    "STAGE5_HALT_TARGET_REPAIR_MODEL_LADDER",
    "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4",
)
HALT_TARGET_NLL_WEIGHT = os.environ.get("STAGE5_HALT_TARGET_REPAIR_NLL_WEIGHT", "0.5")
PHASE1_STEPS = os.environ.get("STAGE5_HALT_TARGET_REPAIR_STEPS", "100")
PHASE1_LR = os.environ.get("STAGE5_HALT_TARGET_REPAIR_LR", "2e-6")
PHASE1_BETA = os.environ.get("STAGE5_HALT_TARGET_REPAIR_BETA", "0.12")
OPTIMIZER_MODULES = os.environ.get("STAGE5_HALT_TARGET_REPAIR_OPTIMIZER_MODULES", "all")


def path_for_cli(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), "".join(chunks), None)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def checkpoint_from_summary(summary_path: str) -> str:
    payload = read_json(summary_path)
    for key in ("phase1_checkpoint", "checkpoint", "selected_checkpoint"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            checkpoint = resolve_path(value)
            if checkpoint.exists():
                return path_for_cli(checkpoint)
    raise FileNotFoundError(f"No local checkpoint found in {summary_path}")


def main() -> int:
    source_summary_path = resolve_path(SOURCE_SUMMARY)
    scored_rows_path = resolve_path(SCORED_ROWS)
    work_dir_path = resolve_path(WORK_DIR)
    if not source_summary_path.exists():
        raise FileNotFoundError(f"Missing source summary: {source_summary_path}")
    if not scored_rows_path.exists():
        raise FileNotFoundError(f"Missing scored rows: {scored_rows_path}")

    checkpoint = checkpoint_from_summary(SOURCE_SUMMARY)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "source_summary": path_for_cli(source_summary_path),
                "source_checkpoint": checkpoint,
                "scored_rows": path_for_cli(scored_rows_path),
                "work_dir": path_for_cli(work_dir_path),
                "halt_target_nll_weight": HALT_TARGET_NLL_WEIGHT,
                "phase1_steps": PHASE1_STEPS,
                "phase1_lr": PHASE1_LR,
                "phase1_beta": PHASE1_BETA,
                "optimizer_modules": OPTIMIZER_MODULES,
            },
            indent=2,
        ),
        flush=True,
    )

    run(
        [
            sys.executable,
            "training/build_capability_ladder_curriculum.py",
            "--input_jsonl",
            path_for_cli(scored_rows_path),
            "--work_dir",
            path_for_cli(work_dir_path),
            "--model_ladder",
            MODEL_LADDER,
            "--allow_answer_only",
            "--assume_decontaminated",
        ]
    )

    env = os.environ.copy()
    env.update(
        {
            "STAGE5_CURRICULUM_SFT_RUN_ID": RUN_ID,
            "STAGE5_CURRICULUM_WORK_DIR": path_for_cli(work_dir_path),
            "STAGE5_CURRICULUM_SUMMARY_JSON": path_for_cli(work_dir_path / "summary.json"),
            "STAGE5_CURRICULUM_RESUME_FROM": checkpoint,
            "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS": "50",
            "STAGE5_CURRICULUM_MIN_MODE_ROWS": "direct=20,deep_narrow=20",
            "STAGE5_CURRICULUM_VAL_FRACTION": "0.10",
            "STAGE5_CURRICULUM_VAL_MIN_ROWS": "1",
            "STAGE5_CURRICULUM_PHASE1_STEPS": PHASE1_STEPS,
            "STAGE5_CURRICULUM_PHASE1_LR": PHASE1_LR,
            "STAGE5_CURRICULUM_PHASE1_BETA": PHASE1_BETA,
            "STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT": HALT_TARGET_NLL_WEIGHT,
            "STAGE5_CURRICULUM_OPTIMIZER_MODULES": OPTIMIZER_MODULES,
            "STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP": "1",
            "STAGE5_CURRICULUM_ALLOW_CROSS_MODEL_ONLY_ANSWERS": "1",
            "STAGE5_CURRICULUM_SFT_DEPTH_GRADIENT_MARGIN": "0.25",
            "STAGE5_CURRICULUM_SFT_REQUIRE_DEPTH_GRADIENT": "1",
            "STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS": "1",
            "STAGE5_CURRICULUM_SFT_PUSH": "1",
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    run([sys.executable, "colab/run_stage5_curriculum_sft.py"], env=env)
    summary = ROOT / "outputs" / "stage5" / RUN_ID / "summary.json"
    print(f"summary_json={path_for_cli(summary)}")
    if summary.exists():
        print(summary.read_text(encoding="utf-8")[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
