"""Check whether duplicated trajectory evaluation preserves Phase1 decisions.

The particle noise sweep found that K=4, noise=0, repulsion=0 scored 69/128
while deterministic Phase1 scored 70/128. Before interpreting particle results,
we need to know whether this is expected bf16/batch-layout drift or a bug in the
multi-trajectory path.
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
RUN_ID = os.environ.get("STAGE4_INVARIANCE_RUN_ID") or time.strftime("stage4_trajectory_invariance_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage4" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

ARC_LIMIT = int(os.environ.get("STAGE4_INVARIANCE_ARC_LIMIT", "128"))
ARC_SPLIT = os.environ.get("STAGE4_INVARIANCE_ARC_SPLIT", "validation")
ARC_SEED = int(os.environ.get("STAGE4_INVARIANCE_ARC_SEED", "0"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE4_INVARIANCE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"
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
        raise FileNotFoundError(f"Missing {PHASE1_CKPT} and Drive restore source {drive_src}")
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


def compare_to_reference(path: Path, ref_path: Path, aggregate: str = "mean") -> dict[str, Any]:
    ref_rows = [row for row in read_jsonl(ref_path) if row["aggregate"] == "mean"]
    test_rows = [row for row in read_jsonl(path) if row["aggregate"] == aggregate]
    ref = {row["id"]: row for row in ref_rows}
    changed: list[dict[str, Any]] = []
    helped = hurt = tied = 0
    score_delta_abs: list[float] = []
    for row in test_rows:
        base = ref[row["id"]]
        if row["prediction"] != base["prediction"]:
            changed.append(
                {
                    "id": row["id"],
                    "reference_prediction": base["prediction"],
                    "test_prediction": row["prediction"],
                    "answer": row["answer"],
                    "reference_hit": base["hit"],
                    "test_hit": row["hit"],
                }
            )
        if row["hit"] and not base["hit"]:
            helped += 1
        elif base["hit"] and not row["hit"]:
            hurt += 1
        else:
            tied += 1
        for label, value in row["scores"].items():
            if label in base["scores"]:
                score_delta_abs.append(abs(float(value) - float(base["scores"][label])))
    return {
        "aggregate": aggregate,
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "prediction_changes": len(changed),
        "changed_examples": changed[:20],
        "mean_abs_score_delta": sum(score_delta_abs) / max(len(score_delta_abs), 1),
        "max_abs_score_delta": max(score_delta_abs) if score_delta_abs else 0.0,
    }


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
        print("No invariance outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 4 trajectory invariance diagnostic"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    restore_from_drive()
    assert PHASE1_CKPT.exists(), PHASE1_CKPT
    (RUN_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "base_run_id": BASE_RUN_ID,
                "arc_limit": ARC_LIMIT,
                "arc_split": ARC_SPLIT,
                "arc_seed": ARC_SEED,
                "phase1_checkpoint": str(PHASE1_CKPT.relative_to(ROOT)),
                "dtype": DTYPE,
                "adapter_dtype": ADAPTER_DTYPE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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

    outputs = {
        "phase1_k1": eval_mcq(
            "phase1_k1",
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
        "phase2path_k1_none": eval_mcq(
            "phase2path_k1_none",
            [
                "--mode",
                "phase2",
                "--checkpoint",
                str(PHASE1_CKPT.relative_to(ROOT)),
                "--max_loops",
                "4",
                "--num_trajectories",
                "1",
                "--aggregate",
                "mean",
                "--particle_update_mode",
                "none",
                "--particle_init_noise",
                "0",
            ],
        ),
        "phase2path_k4_none": eval_mcq(
            "phase2path_k4_none",
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
                "none",
                "--particle_init_noise",
                "0",
            ],
        ),
        "phase2path_k4_svgd0": eval_mcq(
            "phase2path_k4_svgd0",
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
                "0",
                "--svgd_repulsion_scale",
                "0",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
    }

    reference = outputs["phase1_k1"]
    summary = {
        label: {
            "summary": summarize_mcq(path),
            "comparison_to_phase1": (
                None
                if label == "phase1_k1"
                else {
                    aggregate: compare_to_reference(path, reference, aggregate)
                    for aggregate in summarize_mcq(path)
                }
            ),
        }
        for label, path in outputs.items()
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Stage 4 Trajectory Invariance - {RUN_ID}", ""]
    for label, item in summary.items():
        lines.append(f"- {label}: `{item}`")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
