"""Run a stricter particle-value gate on the recovered Stage 4 recurrent model.

This is deliberately not a training run. It asks whether particles/SVGD add
value *at all* over the recovered deterministic recurrent checkpoint before we
spend more A100 time training Phase2.

The earlier Stage 4 sweeps found that particle arms underperformed Phase1, but
the K=4 no-noise control also drifted in bfloat16. The precision probe showed
that the K=4 control is invariant in float32. This runner therefore:

1. keeps the deterministic Phase1 baseline visible,
2. checks K=4/no-noise invariance in bfloat16 and float32,
3. screens low-noise particle and SVGD arms in float32,
4. compares helped/hurt/tied examples against Phase1.

Gate:
    Continue particle training only if a float32 particle arm is non-negative
    against Phase1 and helped examples >= hurt examples.
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
RUN_ID = os.environ.get("STAGE4_GATE_RUN_ID") or time.strftime("stage4_particle_value_gate_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage4" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

ARC_LIMIT = int(os.environ.get("STAGE4_GATE_ARC_LIMIT", "128"))
ARC_SPLIT = os.environ.get("STAGE4_GATE_ARC_SPLIT", "validation")
ARC_SEED = int(os.environ.get("STAGE4_GATE_ARC_SEED", "0"))
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE4_GATE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

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
        raise FileNotFoundError(f"Missing local Stage 4 artifacts and Drive restore source {drive_src}")
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


def helped_hurt(variant_path: Path, phase1_path: Path, aggregate: str) -> dict[str, int]:
    phase1 = {row["id"]: row for row in read_jsonl(phase1_path) if row["aggregate"] == "mean"}
    helped = hurt = tied = changed = 0
    for row in read_jsonl(variant_path):
        if row["aggregate"] != aggregate:
            continue
        base = phase1[row["id"]]
        base_hit = bool(base["hit"])
        hit = bool(row["hit"])
        if row["prediction"] != base["prediction"]:
            changed += 1
        if hit and not base_hit:
            helped += 1
        elif base_hit and not hit:
            hurt += 1
        else:
            tied += 1
    return {"helped": helped, "hurt": hurt, "tied": tied, "prediction_changes": changed}


def prepare_arc() -> None:
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


def calibrate_phase1_projection() -> None:
    if PHASE1_PROJECTION.exists() and PHASE1_PROJECTION.with_suffix(".json").exists():
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
            "bfloat16",
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name="calibrate_phase1_projection.log",
    )


def eval_mcq(label: str, dtype: str, args: list[str]) -> Path:
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
            dtype,
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


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No particle-value gate outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 4 particle value gate"])
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

    prepare_arc()
    calibrate_phase1_projection()

    common_phase1 = [
        "--checkpoint",
        str(PHASE1_CKPT.relative_to(ROOT)),
        "--max_loops",
        "4",
    ]
    common_particles = [
        "--mode",
        "phase2",
        *common_phase1,
        "--num_trajectories",
        "4",
        "--aggregates",
        "mean,max,vote",
    ]
    common_projected_svgd = [
        "--particle_update_mode",
        "svgd",
        "--svgd_kernel_geometry",
        "euclidean",
        "--svgd_kernel_projection_path",
        str(PHASE1_PROJECTION.relative_to(ROOT)),
        "--svgd_kernel_projection_dim",
        "8",
        "--svgd_repulsion_max_norm",
        "none",
    ]

    jobs: list[tuple[str, str, list[str]]] = [
        ("base_bfloat16", "bfloat16", ["--mode", "base", "--aggregate", "mean"]),
        (
            "phase1_k1_bfloat16",
            "bfloat16",
            ["--mode", "phase1", *common_phase1, "--num_trajectories", "1", "--aggregate", "mean"],
        ),
        (
            "phase1_k1_float32",
            "float32",
            ["--mode", "phase1", *common_phase1, "--num_trajectories", "1", "--aggregate", "mean"],
        ),
        (
            "phase1_k4_none_bfloat16",
            "bfloat16",
            [*common_particles, "--particle_update_mode", "none", "--particle_init_noise", "0"],
        ),
        (
            "phase1_k4_none_float32",
            "float32",
            [*common_particles, "--particle_update_mode", "none", "--particle_init_noise", "0"],
        ),
        (
            "phase1_k4_noise005_rep0_float32",
            "float32",
            [
                *common_particles,
                *common_projected_svgd,
                "--particle_init_noise",
                "0.005",
                "--svgd_repulsion_scale",
                "0",
            ],
        ),
        (
            "phase1_k4_noise01_rep0_float32",
            "float32",
            [
                *common_particles,
                *common_projected_svgd,
                "--particle_init_noise",
                "0.01",
                "--svgd_repulsion_scale",
                "0",
            ],
        ),
        (
            "phase1_k4_noise005_rep05_float32",
            "float32",
            [
                *common_particles,
                *common_projected_svgd,
                "--particle_init_noise",
                "0.005",
                "--svgd_repulsion_scale",
                "0.5",
            ],
        ),
        (
            "phase1_k4_noise01_rep05_float32",
            "float32",
            [
                *common_particles,
                *common_projected_svgd,
                "--particle_init_noise",
                "0.01",
                "--svgd_repulsion_scale",
                "0.5",
            ],
        ),
        (
            "phase1_k4_noise005_rep2_float32",
            "float32",
            [
                *common_particles,
                *common_projected_svgd,
                "--particle_init_noise",
                "0.005",
                "--svgd_repulsion_scale",
                "2",
            ],
        ),
        (
            "phase2_k1_float32",
            "float32",
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE2_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "1",
                "--aggregate",
                "mean",
            ],
        ),
        (
            "phase2_k4_svgd_float32",
            "float32",
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
                "0.005",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(STAGE4_PROJECTION.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "0.5",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
    ]

    outputs: dict[str, Path] = {}
    for label, dtype, args in jobs:
        print(f"\n\n==== {label} dtype={dtype} ====")
        outputs[label] = eval_mcq(label, dtype, args)

    summaries = {label: summarize_mcq(path) for label, path in outputs.items()}
    phase1_ref = outputs["phase1_k1_float32"]
    phase1_acc = mean_accuracy(summaries["phase1_k1_float32"])
    base_acc = mean_accuracy(summaries["base_bfloat16"])

    rows: dict[str, Any] = {}
    for label, summary in summaries.items():
        best = best_accuracy(summary)
        comparison = {}
        for aggregate in summary:
            comparison[aggregate] = helped_hurt(outputs[label], phase1_ref, aggregate)
        rows[label] = {
            "summary": summary,
            "best_accuracy": best,
            "lift_over_phase1_float32": None if best is None or phase1_acc is None else best - phase1_acc,
            "gap_to_base_bfloat16": None if best is None or base_acc is None else best - base_acc,
            "comparison_to_phase1_float32": comparison,
        }

    result = {"metadata": metadata, "rows": rows}
    (RUN_DIR / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 4 Particle Value Gate - {RUN_ID}",
        "",
        "## Question",
        "Do particles/SVGD add value over the recovered deterministic recurrent checkpoint before more Phase2 training?",
        "",
        "## Results",
    ]
    for label, info in rows.items():
        lines.append(
            f"- {label}: best={info['best_accuracy']} "
            f"lift_over_phase1_float32={info['lift_over_phase1_float32']} "
            f"gap_to_base_bfloat16={info['gap_to_base_bfloat16']} raw={info['summary']}"
        )
        if label != "phase1_k1_float32":
            lines.append(f"  - helped/hurt/tied={info['comparison_to_phase1_float32']}")

    particle_labels = [
        label
        for label in rows
        if label.startswith("phase1_k4_noise") or label.startswith("phase2_k4")
    ]
    passing = []
    for label in particle_labels:
        info = rows[label]
        best = info["best_accuracy"]
        if best is None or phase1_acc is None or best < phase1_acc:
            continue
        aggregate = max(info["summary"], key=lambda name: info["summary"][name]["accuracy"])
        comparison = info["comparison_to_phase1_float32"][aggregate]
        if comparison["helped"] >= comparison["hurt"]:
            passing.append({"label": label, "aggregate": aggregate, "comparison": comparison})

    lines.extend(
        [
            "",
            "## Gate",
            f"phase1_float32_accuracy={phase1_acc}",
            f"passing_particle_arms={passing}",
            "",
            "Decision: continue particle training only if `passing_particle_arms` is non-empty.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))

    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
