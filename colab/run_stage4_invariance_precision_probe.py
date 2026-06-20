"""Probe whether K=4 no-noise trajectory drift is bf16 numerical drift.

This filters the ARC examples that changed between deterministic Phase1 K=1 and
duplicated K=4/no-update evaluation, then reruns those examples in bfloat16 and
float32. If float32 removes the decision changes, the issue is batch-shape
numerics. If float32 still changes them, investigate the trajectory wrapper.
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
INVARIANCE_RUN_ID = os.environ.get("STAGE4_INVARIANCE_RUN_ID", "stage4_trajectory_invariance_a100_20260620")
RUN_ID = os.environ.get("STAGE4_PRECISION_RUN_ID") or time.strftime("stage4_precision_probe_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage4" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

ARC_SPLIT = os.environ.get("STAGE4_PRECISION_ARC_SPLIT", "validation")
ARC_LIMIT = int(os.environ.get("STAGE4_PRECISION_ARC_LIMIT", "128"))
ARC_SEED = int(os.environ.get("STAGE4_PRECISION_ARC_SEED", "0"))
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE4_PRECISION_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"
INVARIANCE_SUMMARY = ROOT / "outputs" / "stage4" / INVARIANCE_RUN_ID / "summary.json"
ARC_JSONL = ROOT / "data" / f"{RUN_ID}_arc{ARC_LIMIT}.jsonl"
PROBE_JSONL = ROOT / "data" / f"{RUN_ID}_probe.jsonl"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def changed_ids() -> list[str]:
    summary = json.loads(INVARIANCE_SUMMARY.read_text(encoding="utf-8"))
    changed = summary["phase2path_k4_none"]["comparison_to_phase1"]["mean"]["changed_examples"]
    return [item["id"] for item in changed]


def summarize(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aggregate.setdefault(row["aggregate"], []).append(row)
    return {
        aggregate: {
            "correct": sum(bool(row["hit"]) for row in aggregate_rows),
            "total": len(aggregate_rows),
            "accuracy": sum(bool(row["hit"]) for row in aggregate_rows) / max(len(aggregate_rows), 1),
        }
        for aggregate, aggregate_rows in sorted(by_aggregate.items())
    }


def compare(test_path: Path, ref_path: Path, aggregate: str) -> dict[str, Any]:
    ref = {row["id"]: row for row in read_jsonl(ref_path) if row["aggregate"] == "mean"}
    test = [row for row in read_jsonl(test_path) if row["aggregate"] == aggregate]
    changes = []
    score_deltas = []
    for row in test:
        base = ref[row["id"]]
        if row["prediction"] != base["prediction"]:
            changes.append(
                {
                    "id": row["id"],
                    "reference_prediction": base["prediction"],
                    "test_prediction": row["prediction"],
                    "answer": row["answer"],
                    "reference_hit": base["hit"],
                    "test_hit": row["hit"],
                }
            )
        for label, value in row["scores"].items():
            score_deltas.append(abs(float(value) - float(base["scores"][label])))
    return {
        "aggregate": aggregate,
        "prediction_changes": len(changes),
        "changed_examples": changes,
        "mean_abs_score_delta": sum(score_deltas) / max(len(score_deltas), 1),
        "max_abs_score_delta": max(score_deltas) if score_deltas else 0.0,
    }


def eval_mcq(label: str, dtype: str, args: list[str]) -> Path:
    output = RUN_DIR / f"{label}_{dtype}.jsonl"
    if output.exists():
        output.unlink()
    run(
        [
            sys.executable,
            "eval/eval_mcq.py",
            "--data_jsonl",
            str(PROBE_JSONL.relative_to(ROOT)),
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
        log_name=f"{label}_{dtype}.log",
    )
    return output


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No precision probe outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 4 trajectory precision probe"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    restore_from_drive()
    assert PHASE1_CKPT.exists(), PHASE1_CKPT
    assert INVARIANCE_SUMMARY.exists(), INVARIANCE_SUMMARY

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
    ids = set(changed_ids())
    rows = [row for row in read_jsonl(ARC_JSONL) if row["id"] in ids]
    write_jsonl(PROBE_JSONL, rows)
    print(f"probe_ids={sorted(ids)}")
    print(f"probe_rows={len(rows)}")
    assert len(rows) == len(ids), (len(rows), ids)

    outputs: dict[str, Path] = {}
    for dtype in ["bfloat16", "float32"]:
        outputs[f"phase1_k1_{dtype}"] = eval_mcq(
            "phase1_k1",
            dtype,
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
        outputs[f"phase2path_k4_none_{dtype}"] = eval_mcq(
            "phase2path_k4_none",
            dtype,
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
        )

    summary: dict[str, Any] = {}
    for dtype in ["bfloat16", "float32"]:
        ref = outputs[f"phase1_k1_{dtype}"]
        test = outputs[f"phase2path_k4_none_{dtype}"]
        summary[dtype] = {
            "phase1": summarize(ref),
            "phase2path_k4_none": summarize(test),
            "comparison": {
                aggregate: compare(test, ref, aggregate)
                for aggregate in summarize(test)
            },
        }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Stage 4 Invariance Precision Probe - {RUN_ID}", ""]
    for dtype, payload in summary.items():
        lines.append(f"- {dtype}: `{payload}`")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
