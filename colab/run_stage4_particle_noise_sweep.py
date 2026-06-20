"""Sweep smaller inference-time particle perturbations on the Stage 4 Phase1 checkpoint.

The previous rescue ablation showed that K=4, noise=0.05 particle updates hurt
ARC-128 relative to deterministic Phase1. This runner asks a narrower question:

    Is there a lower-noise/K regime that preserves Phase1 and maybe converts
    some of the helped examples without adding as many harmed examples?

It also includes a zero-noise K=4 control. That control should match Phase1; if
it does not, the multi-trajectory likelihood path itself needs debugging.
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
RUN_ID = os.environ.get("STAGE4_NOISE_RUN_ID") or time.strftime("stage4_particle_noise_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage4" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

ARC_LIMIT = int(os.environ.get("STAGE4_NOISE_ARC_LIMIT", "128"))
ARC_SPLIT = os.environ.get("STAGE4_NOISE_ARC_SPLIT", "validation")
ARC_SEED = int(os.environ.get("STAGE4_NOISE_ARC_SEED", "0"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE4_NOISE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"
PHASE1_PROJECTION = RUN_DIR / "phase1_within_group_projection.pt"
ARC_JSONL = ROOT / "data" / f"{RUN_ID}_arc{ARC_LIMIT}.jsonl"

VARIANTS = [
    {"label": "phase1_k4_noise0_rep0", "k": 4, "noise": 0.0, "repulsion": 0.0},
    {"label": "phase1_k2_noise005_rep05", "k": 2, "noise": 0.005, "repulsion": 0.5},
    {"label": "phase1_k2_noise01_rep05", "k": 2, "noise": 0.01, "repulsion": 0.5},
    {"label": "phase1_k4_noise005_rep05", "k": 4, "noise": 0.005, "repulsion": 0.5},
    {"label": "phase1_k4_noise01_rep05", "k": 4, "noise": 0.01, "repulsion": 0.5},
    {"label": "phase1_k4_noise02_rep05", "k": 4, "noise": 0.02, "repulsion": 0.5},
    {"label": "phase1_k4_noise005_rep2", "k": 4, "noise": 0.005, "repulsion": 2.0},
]


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
    if PHASE1_CKPT.exists():
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
            f"Missing local Phase1 checkpoint under {BASE_RUN_DIR} and Drive restore source {drive_src}"
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


def helped_hurt(variant_path: Path, phase1_path: Path, aggregate: str) -> dict[str, int]:
    phase1 = {row["id"]: row for row in read_jsonl(phase1_path)}
    variant = [row for row in read_jsonl(variant_path) if row["aggregate"] == aggregate]
    helped = hurt = tied = 0
    for row in variant:
        base_hit = bool(phase1[row["id"]]["hit"])
        hit = bool(row["hit"])
        if hit and not base_hit:
            helped += 1
        elif base_hit and not hit:
            hurt += 1
        else:
            tied += 1
    return {"helped": helped, "hurt": hurt, "tied": tied}


def calibrate_projection() -> None:
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
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name="calibrate_projection.log",
    )


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


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No noise sweep outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 4 particle noise sweep"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    restore_from_drive()
    assert PHASE1_CKPT.exists(), PHASE1_CKPT

    metadata = {
        "run_id": RUN_ID,
        "base_run_id": BASE_RUN_ID,
        "arc_limit": ARC_LIMIT,
        "arc_split": ARC_SPLIT,
        "arc_seed": ARC_SEED,
        "phase1_checkpoint": str(PHASE1_CKPT.relative_to(ROOT)),
        "variants": VARIANTS,
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
    calibrate_projection()

    outputs: dict[str, Path] = {}
    outputs["phase1_label"] = eval_mcq(
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
    )

    for variant in VARIANTS:
        label = variant["label"]
        outputs[label] = eval_mcq(
            label,
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE1_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                str(variant["k"]),
                "--aggregates",
                "mean,max,vote",
                "--particle_update_mode",
                "svgd",
                "--particle_init_noise",
                str(variant["noise"]),
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(PHASE1_PROJECTION.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                str(variant["repulsion"]),
                "--svgd_repulsion_max_norm",
                "none",
            ],
        )

    summaries = {label: summarize_mcq(path) for label, path in outputs.items()}
    phase1_acc = mean_accuracy(summaries["phase1_label"])
    table: dict[str, Any] = {}
    for label, summary in summaries.items():
        best = best_accuracy(summary)
        row: dict[str, Any] = {
            "summary": summary,
            "best_accuracy": best,
            "lift_over_phase1": delta(best, phase1_acc),
        }
        if label != "phase1_label":
            row["helped_hurt"] = {
                aggregate: helped_hurt(outputs[label], outputs["phase1_label"], aggregate)
                for aggregate in summary
            }
        table[label] = row

    result = {"metadata": metadata, "table": table}
    (RUN_DIR / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 4 Particle Noise Sweep - {RUN_ID}",
        "",
        f"Phase1 deterministic baseline: `{phase1_acc}`",
        "",
        "## Results",
    ]
    for label, row in table.items():
        lines.append(
            f"- {label}: best={row['best_accuracy']} lift={row['lift_over_phase1']} "
            f"raw={row['summary']} helped_hurt={row.get('helped_hurt')}"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "A particle setting must match or beat Phase1 and have helped >= hurt before we train around it.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))

    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
