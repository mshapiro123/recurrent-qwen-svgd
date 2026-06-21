"""Run particle/SVGD ARC gates from the recovered deterministic Phase 1 parent.

This answers the next question after deterministic recovery:

    Once Phase 1 recurrent matches base Qwen, do K=4 particles add value?

The script restores the recovered Phase 1 checkpoint from Drive if necessary,
calibrates a within-group SVGD projection for that checkpoint, then launches
ARC-Challenge benchmark-suite runs for a small particle ladder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    DEFAULT_CHECKPOINT_REL,
    DEFAULT_RECOVERED_RUN_ID,
    DEFAULT_SOURCE_SUMMARY_REL,
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_RECOVERED_PARTICLE_RUN_ID") or time.strftime(
    "stage5_recovered_phase1_particles_arc_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

RECOVERED_RUN_ID = os.environ.get("STAGE5_RECOVERED_PHASE1_RUN_ID", DEFAULT_RECOVERED_RUN_ID)
CHECKPOINT = Path(os.environ.get("STAGE5_RECOVERED_PHASE1_CHECKPOINT", DEFAULT_CHECKPOINT_REL))
if not CHECKPOINT.is_absolute():
    CHECKPOINT = ROOT / CHECKPOINT
SOURCE_SUMMARY = Path(os.environ.get("STAGE5_RECOVERED_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY_REL))
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

ARC_LIMIT = os.environ.get("STAGE5_RECOVERED_PARTICLE_ARC_LIMIT", "256")
NUM_TRAJECTORIES = int(os.environ.get("STAGE5_RECOVERED_PARTICLE_K", "4"))
PROJECTION_DIM = os.environ.get("STAGE5_RECOVERED_PARTICLE_PROJECTION_DIM", "8")
PARTICLE_INIT_NOISE = os.environ.get("STAGE5_RECOVERED_PARTICLE_INIT_NOISE", "0.05")
REPULSION_SCALES = os.environ.get("STAGE5_RECOVERED_PARTICLE_REPULSIONS", "0,2")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_RECOVERED_PARTICLE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


@dataclass(frozen=True)
class ParticleArm:
    name: str
    repulsion_scale: str


def parse_repulsion_arms(value: str) -> list[ParticleArm]:
    arms: list[ParticleArm] = []
    for raw in value.split(","):
        scale = raw.strip()
        if not scale:
            continue
        label = scale.replace(".", "p").replace("-", "m")
        arms.append(ParticleArm(name=f"rep{label}", repulsion_scale=scale))
    return arms


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
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def projection_path() -> Path:
    return RUN_DIR / "recovered_phase1_within_group_projection.pt"


def calibrate_projection() -> Path:
    output = projection_path()
    if output.exists() and output.with_suffix(".json").exists():
        return output
    run(
        [
            sys.executable,
            "eval/calibrate_svgd_projection.py",
            "--tasks_jsonl",
            "eval/smoke_exact_tasks_v2.jsonl",
            "--phase2_checkpoint",
            path_for_cli(CHECKPOINT),
            "--seeds",
            "0,1,2,3,4",
            "--num_trajectories",
            str(NUM_TRAJECTORIES),
            "--particle_init_noise",
            PARTICLE_INIT_NOISE,
            "--svgd_repulsion_scale",
            "1.0",
            "--svgd_repulsion_max_norm",
            "none",
            "--calibration_centering",
            "within_group",
            "--projection_dim",
            "64",
            "--output",
            path_for_cli(output),
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name="calibrate_projection.log",
    )
    return output


def benchmark_arm(arm: ParticleArm, projection: Path) -> Path:
    run_id = f"{RUN_ID}_{arm.name}_k{NUM_TRAJECTORIES}_arc{ARC_LIMIT}"
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
            "STAGE5_BENCHMARK_CHECKPOINT": path_for_cli(CHECKPOINT),
            "STAGE5_BENCHMARKS": "arc_challenge",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": ARC_LIMIT,
            "STAGE5_BENCHMARK_SCORE_TARGETS": "label",
            "STAGE5_BENCHMARK_AGGREGATES": "mean,max,vote",
            "STAGE5_BENCHMARK_RECURRENT_MODE": "phase2",
            "STAGE5_BENCHMARK_NUM_TRAJECTORIES": str(NUM_TRAJECTORIES),
            "STAGE5_BENCHMARK_PARTICLE_UPDATE_MODE": "svgd",
            "STAGE5_BENCHMARK_PARTICLE_INIT_NOISE": PARTICLE_INIT_NOISE,
            "STAGE5_BENCHMARK_SVGD_REPULSION_SCALE": arm.repulsion_scale,
            "STAGE5_BENCHMARK_SVGD_REPULSION_MAX_NORM": "none",
            "STAGE5_BENCHMARK_SVGD_KERNEL_PROJECTION_PATH": path_for_cli(projection),
            "STAGE5_BENCHMARK_SVGD_KERNEL_PROJECTION_DIM": PROJECTION_DIM,
            "STAGE5_BENCHMARK_SVGD_KERNEL_GEOMETRY": "euclidean",
            "STAGE5_BENCHMARK_CONTINUE_ON_FAILURE": "0",
            "STAGE5_BENCHMARK_PUSH": "1" if PUSH_RESULTS else "0",
            "DTYPE": DTYPE,
            "ADAPTER_DTYPE": ADAPTER_DTYPE,
            "DEVICE": DEVICE,
        }
    )
    if SOURCE_SUMMARY.exists():
        env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = path_for_cli(SOURCE_SUMMARY)
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env, log_name=f"{arm.name}_benchmark.log")
    return ROOT / "outputs" / "stage5" / run_id / "summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_result(summary_path: Path) -> dict[str, Any]:
    payload = read_json(summary_path)
    paired = payload["paired_comparisons"]["arc_challenge"]["label"]
    comparisons = payload["comparisons"]["arc_challenge"]["label"]
    return {
        "summary_path": path_for_cli(summary_path),
        "checkpoint": payload.get("checkpoint"),
        "recurrent_num_trajectories": payload.get("recurrent_num_trajectories"),
        "comparisons": comparisons,
        "paired_comparisons": paired,
    }


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No recovered particle gate outputs changed.")
        return
    run(["git", "commit", "-m", f"Record recovered Phase1 particle ARC gate {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def write_summary(arms: list[ParticleArm], summaries: dict[str, Path], projection: Path) -> None:
    results = {arm.name: compact_result(summaries[arm.name]) for arm in arms}
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_recovered_phase1_particle_arc_gate",
        "arc_limit": ARC_LIMIT,
        "checkpoint": path_for_cli(CHECKPOINT),
        "projection": path_for_cli(projection),
        "projection_dim": PROJECTION_DIM,
        "num_trajectories": NUM_TRAJECTORIES,
        "particle_init_noise": PARTICLE_INIT_NOISE,
        "results": results,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Recovered Phase1 Particle ARC Gate - {RUN_ID}",
        "",
        f"- Checkpoint: `{path_for_cli(CHECKPOINT)}`",
        f"- ARC limit: `{ARC_LIMIT}`",
        f"- K: `{NUM_TRAJECTORIES}`",
        f"- Projection: `{path_for_cli(projection)}`",
        "",
        "## Arms",
    ]
    for arm in arms:
        mean = results[arm.name]["comparisons"]["mean"]
        paired_mean = results[arm.name]["paired_comparisons"]["mean"]
        max_row = results[arm.name]["comparisons"].get("max")
        vote_row = results[arm.name]["comparisons"].get("vote")
        lines.append(
            f"- `{arm.name}` repulsion `{arm.repulsion_scale}`: mean recurrent "
            f"`{mean['recurrent']['correct']}/{mean['recurrent']['total']}` vs base "
            f"`{mean['base']['correct']}/{mean['base']['total']}`, delta "
            f"`{mean['correct_delta_recurrent_vs_base']}`, W/L/T "
            f"`{paired_mean['wins']}/{paired_mean['losses']}/{paired_mean['ties']}`, p "
            f"`{paired_mean['sign_test_p_value']}`"
        )
        if max_row and vote_row:
            lines.append(
                f"  - max `{max_row['recurrent']['correct']}/{max_row['recurrent']['total']}`, "
                f"vote `{vote_row['recurrent']['correct']}/{vote_row['recurrent']['total']}`"
            )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    restore_checkpoint_if_needed(CHECKPOINT, run_id=RECOVERED_RUN_ID)
    projection = calibrate_projection()
    arms = parse_repulsion_arms(REPULSION_SCALES)
    if not arms:
        raise ValueError("No particle arms requested.")
    (RUN_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "recovered_run_id": RECOVERED_RUN_ID,
                "checkpoint": path_for_cli(CHECKPOINT),
                "source_summary": path_for_cli(SOURCE_SUMMARY) if SOURCE_SUMMARY.exists() else None,
                "arc_limit": ARC_LIMIT,
                "num_trajectories": NUM_TRAJECTORIES,
                "particle_init_noise": PARTICLE_INIT_NOISE,
                "repulsion_scales": [arm.repulsion_scale for arm in arms],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summaries = {arm.name: benchmark_arm(arm, projection) for arm in arms}
    write_summary(arms, summaries, projection)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
