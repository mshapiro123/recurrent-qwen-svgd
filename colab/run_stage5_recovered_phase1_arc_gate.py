"""Run the recovered Phase 1 ARC-Challenge gate.

This wrapper exists to make the next Colab step one command:

    python colab/run_stage5_recovered_phase1_arc_gate.py

It benchmarks the best recovered deterministic recurrent checkpoint against
unmodified Qwen on ARC-Challenge, defaulting to a 256-row validation slice. The
checkpoint itself is not committed to git, so the wrapper restores it from the
Drive artifact backup when needed before delegating to
``run_stage5_benchmark_suite.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_RECOVERED_RUN_ID = "stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231"
DEFAULT_CHECKPOINT_REL = (
    f"outputs/stage5/{DEFAULT_RECOVERED_RUN_ID}/phase1/phase1_step_125.pt"
)
DEFAULT_SOURCE_SUMMARY_REL = f"outputs/stage5/{DEFAULT_RECOVERED_RUN_ID}/summary.json"


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mount_drive_if_possible() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}")


def drive_root() -> Path:
    return Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))


def candidate_drive_checkpoints(run_id: str, filename: str) -> list[Path]:
    root = drive_root()
    candidates = [
        root / run_id / "run_dir" / "phase1" / filename,
        root / run_id / "phase1" / filename,
    ]
    if root.exists():
        candidates.extend(sorted(root.glob(f"{run_id}*/run_dir/phase1/{filename}")))
        candidates.extend(sorted(path for path in root.rglob(filename) if run_id in str(path)))
    return candidates


def restore_checkpoint_if_needed(checkpoint: Path, *, run_id: str) -> None:
    if checkpoint.exists():
        print(f"checkpoint_present={path_for_cli(checkpoint)}")
        return
    mount_drive_if_possible()
    for candidate in candidate_drive_checkpoints(run_id, checkpoint.name):
        if candidate.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, checkpoint)
            print(f"restored_checkpoint={candidate} -> {checkpoint}")
            return
    searched = "\n".join(str(path) for path in candidate_drive_checkpoints(run_id, checkpoint.name)[:12])
    raise FileNotFoundError(
        f"Missing recovered checkpoint {checkpoint}. Could not restore it from Drive.\n"
        f"Drive root: {drive_root()}\nSearched:\n{searched}"
    )


def run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
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
    if proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc


def default_run_id(limit: str) -> str:
    limit_label = limit.strip().lower().replace("none", "full").replace("all", "full")
    return f"stage5_recovered_phase1_arc{limit_label}_{time.strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    recovered_run_id = os.environ.get("STAGE5_RECOVERED_PHASE1_RUN_ID", DEFAULT_RECOVERED_RUN_ID)
    checkpoint = Path(os.environ.get("STAGE5_RECOVERED_PHASE1_CHECKPOINT", DEFAULT_CHECKPOINT_REL))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    limit = os.environ.get("STAGE5_RECOVERED_ARC_LIMIT", "256")
    source_summary = Path(os.environ.get("STAGE5_RECOVERED_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY_REL))
    if not source_summary.is_absolute():
        source_summary = ROOT / source_summary

    restore_checkpoint_if_needed(checkpoint, run_id=recovered_run_id)

    env = os.environ.copy()
    env.setdefault("STAGE5_BENCHMARK_SUITE_RUN_ID", default_run_id(limit))
    env["STAGE5_BENCHMARK_CHECKPOINT"] = path_for_cli(checkpoint)
    if source_summary.exists():
        env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env["STAGE5_BENCHMARKS"] = "arc_challenge"
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = limit
    env["STAGE5_BENCHMARK_SCORE_TARGETS"] = os.environ.get("STAGE5_BENCHMARK_SCORE_TARGETS", "label")
    env["STAGE5_BENCHMARK_AGGREGATES"] = os.environ.get("STAGE5_BENCHMARK_AGGREGATES", "mean")
    env["STAGE5_BENCHMARK_RECURRENT_MODE"] = "phase1"
    env["STAGE5_BENCHMARK_NUM_TRAJECTORIES"] = "1"
    env.setdefault("STAGE5_BENCHMARK_CONTINUE_ON_FAILURE", "0")
    env.setdefault("STAGE5_BENCHMARK_PUSH", "1")
    env.setdefault("DTYPE", "bfloat16")
    env.setdefault("ADAPTER_DTYPE", "float32")
    env.setdefault("DEVICE", "cuda")

    print(f"recovered_run_id={recovered_run_id}")
    print(f"checkpoint={path_for_cli(checkpoint)}")
    print(f"arc_limit={limit}")
    print(f"benchmark_run_id={env['STAGE5_BENCHMARK_SUITE_RUN_ID']}")
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
