"""Run Phase2 rescue ablations from the strong Stage 4 Phase1 checkpoint.

Goal:
    Separate three effects on the current recurrent baseline:

    1. Base Qwen gap.
    2. Deterministic Phase1 recurrent competence.
    3. Inference-time particle/SVGD lift on top of Phase1, without Phase2
       stochastic training.

This should be run before spending more A100 time on Phase2 training. If
Phase1+particles does not beat Phase1, training Phase2 harder is likely the
wrong next move.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BASE_RUN_ID = os.environ.get("STAGE4_BASE_RUN_ID", "stage4_opus_a100_20260620")
RUN_ID = os.environ.get("STAGE4_RESCUE_RUN_ID") or time.strftime("stage4_phase2_rescue_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage4" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ARC_LIMIT = int(os.environ.get("STAGE4_RESCUE_ARC_LIMIT", "128"))
ARC_SPLIT = os.environ.get("STAGE4_RESCUE_ARC_SPLIT", "validation")
ARC_SEED = int(os.environ.get("STAGE4_RESCUE_ARC_SEED", "0"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE4_RESCUE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"
PHASE2_CKPT = BASE_RUN_DIR / "phase2" / "phase2_step_100.pt"
STAGE4_PROJECTION = BASE_RUN_DIR / "within_group_projection.pt"
PHASE1_PROJECTION = RUN_DIR / "phase1_within_group_projection.pt"
ARC_JSONL = ROOT / "data" / f"{RUN_ID}_arc{ARC_LIMIT}.jsonl"


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def restore_from_drive() -> None:
    """Restore Stage 4 artifacts from Drive if the runtime is fresh."""

    if PHASE1_CKPT.exists() and PHASE2_CKPT.exists() and STAGE4_PROJECTION.exists():
        return
    drive_src = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts") / BASE_RUN_ID / "run_dir"
    if not drive_src.exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Colab only
            print(f"Drive mount failed: {exc}")
    if not drive_src.exists():
        raise FileNotFoundError(
            f"Missing local artifacts under {BASE_RUN_DIR} and Drive restore source {drive_src}"
        )
    BASE_RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(drive_src, BASE_RUN_DIR, dirs_exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def best_accuracy(summary: dict[str, Any]) -> float | None:
    values = [float(item["accuracy"]) for item in summary.values() if item]
    return max(values) if values else None


def mean_accuracy(summary: dict[str, Any]) -> float | None:
    metric = summary.get("mean")
    return float(metric["accuracy"]) if metric else None


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def eval_mcq(label: str, args: list[str]) -> Path:
    output = RUN_DIR / f"{label}.jsonl"
    if output.exists():
        output.unlink()
    run(
        [
            sys.executable,
            "eval/eval_mcq.py",
            "--data_jsonl",
            str(ARC_JSONL.relative_to(ROOT)),
            "--prompt_style",
            "with_options",
            "--score_target",
            "label",
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--seed",
            "0",
            *args,
            "--output_jsonl",
            str(output.relative_to(ROOT)),
        ],
        log_name=f"{label}.log",
    )
    return output


def calibrate_phase1_projection() -> None:
    if PHASE1_PROJECTION.exists() and PHASE1_PROJECTION.with_suffix(".json").exists():
        print(f"phase1_projection_exists={PHASE1_PROJECTION}")
        return
    run(
        [
            sys.executable,
            "eval/calibrate_svgd_projection.py",
            "--tasks_jsonl",
            "eval/smoke_exact_tasks_v2.jsonl",
            "--phase2_checkpoint",
            str(PHASE1_CKPT.relative_to(ROOT)),
            "--seeds",
            "0,1,2,3,4",
            "--num_trajectories",
            "4",
            "--particle_init_noise",
            "0.05",
            "--svgd_repulsion_scale",
            "1.0",
            "--svgd_repulsion_max_norm",
            "none",
            "--calibration_centering",
            "within_group",
            "--projection_dim",
            "64",
            "--output",
            str(PHASE1_PROJECTION.relative_to(ROOT)),
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name="calibrate_phase1_projection.log",
    )


def git_commit_results() -> None:
    patterns = [
        "*.json",
        "*.md",
        "*.log",
        "*.jsonl",
    ]
    for pattern in patterns:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No rescue outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 4 Phase2 rescue ablation"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    restore_from_drive()
    for artifact in [PHASE1_CKPT, PHASE2_CKPT, STAGE4_PROJECTION]:
        print(f"artifact {artifact.relative_to(ROOT)} exists={artifact.exists()}")
        assert artifact.exists(), artifact

    metadata = {
        "run_id": RUN_ID,
        "base_run_id": BASE_RUN_ID,
        "arc_limit": ARC_LIMIT,
        "arc_split": ARC_SPLIT,
        "arc_seed": ARC_SEED,
        "phase1_checkpoint": str(PHASE1_CKPT.relative_to(ROOT)),
        "phase2_checkpoint": str(PHASE2_CKPT.relative_to(ROOT)),
        "stage4_projection": str(STAGE4_PROJECTION.relative_to(ROOT)),
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            "ARC-Challenge",
            "--split",
            ARC_SPLIT,
            "--limit",
            str(ARC_LIMIT),
            "--seed",
            str(ARC_SEED),
            "--output_jsonl",
            str(ARC_JSONL.relative_to(ROOT)),
        ],
        log_name="prepare_arc.log",
    )
    calibrate_phase1_projection()

    jobs: list[tuple[str, list[str]]] = [
        ("base_label", ["--mode", "base", "--aggregate", "mean"]),
        (
            "phase1_label",
            [
                "--mode",
                "phase1",
                "--checkpoint",
                str(PHASE1_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "1",
                "--aggregate",
                "mean",
            ],
        ),
        (
            "phase1_particles_rep0_label",
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE1_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "4",
                "--aggregates",
                "mean,max,vote",
                "--particle_update_mode",
                "svgd",
                "--particle_init_noise",
                "0.05",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(PHASE1_PROJECTION.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "0",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
        (
            "phase1_particles_rep05_label",
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE1_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "4",
                "--aggregates",
                "mean,max,vote",
                "--particle_update_mode",
                "svgd",
                "--particle_init_noise",
                "0.05",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(PHASE1_PROJECTION.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "0.5",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
        (
            "phase1_particles_rep2_label",
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE1_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "4",
                "--aggregates",
                "mean,max,vote",
                "--particle_update_mode",
                "svgd",
                "--particle_init_noise",
                "0.05",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(PHASE1_PROJECTION.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "2",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
        (
            "phase2_stage4_rep2_label",
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE2_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "4",
                "--aggregates",
                "mean,max,vote",
                "--particle_update_mode",
                "svgd",
                "--particle_init_noise",
                "0.05",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(STAGE4_PROJECTION.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "2",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
    ]

    outputs: dict[str, Path] = {}
    for label, args in jobs:
        print(f"\n\n==== {label} ====")
        outputs[label] = eval_mcq(label, args)

    summaries = {label: summarize_mcq(path) for label, path in outputs.items()}
    phase1_acc = mean_accuracy(summaries["phase1_label"])
    base_acc = mean_accuracy(summaries["base_label"])
    ladder = {
        label: {
            "summary": summary,
            "best_accuracy": best_accuracy(summary),
            "lift_over_phase1": delta(best_accuracy(summary), phase1_acc),
            "gap_to_base": delta(best_accuracy(summary), base_acc),
        }
        for label, summary in summaries.items()
    }
    result = {
        "metadata": metadata,
        "summaries": summaries,
        "ladder": ladder,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 4 Phase2 Rescue Ablation - {RUN_ID}",
        "",
        "## Question",
        "Does inference-time particle/SVGD improve the strong Stage 4 Phase1 recurrent baseline before more Phase2 training?",
        "",
        "## ARC Label-Likelihood",
    ]
    for label, info in ladder.items():
        lines.append(
            f"- {label}: best={info['best_accuracy']} lift_over_phase1={info['lift_over_phase1']} "
            f"gap_to_base={info['gap_to_base']} raw={info['summary']}"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "If no Phase1-particle arm beats `phase1_label`, do not continue Phase2 training with the current particle mechanism.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))

    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
