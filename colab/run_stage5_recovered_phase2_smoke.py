"""Train a bounded Phase 2 particle smoke from the recovered Phase 1 parent.

This is the first training check after deterministic recovery reached base
parity. It keeps the run intentionally small:

* restore the recovered Phase 1 checkpoint from Drive if needed;
* calibrate a within-group SVGD projection for that checkpoint;
* train K=4 projected-SVGD Phase 2 for a small number of Opus steps;
* benchmark the resulting checkpoint on ARC-Challenge.

The produced Phase 2 checkpoint is backed up to Drive, while lightweight logs
and summaries are committed to git.
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

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    DEFAULT_CHECKPOINT_REL,
    DEFAULT_RECOVERED_RUN_ID,
    DEFAULT_SOURCE_SUMMARY_REL,
    drive_root,
    mount_drive_if_possible,
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_RECOVERED_PHASE2_RUN_ID") or time.strftime(
    "stage5_recovered_phase2_smoke_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
RECOVERED_RUN_ID = os.environ.get("STAGE5_RECOVERED_PHASE1_RUN_ID", DEFAULT_RECOVERED_RUN_ID)
RECOVERED_CHECKPOINT = Path(os.environ.get("STAGE5_RECOVERED_PHASE1_CHECKPOINT", DEFAULT_CHECKPOINT_REL))
if not RECOVERED_CHECKPOINT.is_absolute():
    RECOVERED_CHECKPOINT = ROOT / RECOVERED_CHECKPOINT
SOURCE_SUMMARY = Path(os.environ.get("STAGE5_RECOVERED_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY_REL))
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

DATASET_ID = os.environ.get("OPUS_DATASET_ID", "lordx64/reasoning-distill-opus-4-7-max-sft")
OPUS_LIMIT = int(os.environ.get("STAGE5_RECOVERED_PHASE2_OPUS_LIMIT", "6000"))
VAL_FRACTION = float(os.environ.get("STAGE5_RECOVERED_PHASE2_VAL_FRACTION", "0.05"))
MAX_TOTAL_TOKENS = int(os.environ.get("STAGE5_RECOVERED_PHASE2_MAX_TOTAL_TOKENS", "1024"))
MAX_LENGTH = int(os.environ.get("STAGE5_RECOVERED_PHASE2_MAX_LENGTH", "512"))
MAX_STEPS = int(os.environ.get("STAGE5_RECOVERED_PHASE2_STEPS", "25"))
SAVE_EVERY = int(os.environ.get("STAGE5_RECOVERED_PHASE2_SAVE_EVERY", str(MAX_STEPS)))
LEARNING_RATE = float(os.environ.get("STAGE5_RECOVERED_PHASE2_LR", "5e-6"))
BETA = float(os.environ.get("STAGE5_RECOVERED_PHASE2_BETA", "0.12"))
MAX_GRAD_NORM = float(os.environ.get("STAGE5_RECOVERED_PHASE2_MAX_GRAD_NORM", "0.3"))
NUM_TRAJECTORIES = int(os.environ.get("STAGE5_RECOVERED_PHASE2_K", "4"))
PARTICLE_INIT_NOISE = os.environ.get("STAGE5_RECOVERED_PHASE2_INIT_NOISE", "0.05")
REPULSION_SCALE = os.environ.get("STAGE5_RECOVERED_PHASE2_REPULSION", "2")
PROJECTION_DIM = os.environ.get("STAGE5_RECOVERED_PHASE2_PROJECTION_DIM", "8")
ARC_LIMIT = os.environ.get("STAGE5_RECOVERED_PHASE2_ARC_LIMIT", "128")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_RECOVERED_PHASE2_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_opus_train.jsonl"
VAL_JSONL = ROOT / "data" / f"{RUN_ID}_opus_val.jsonl"


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


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare_opus() -> None:
    if TRAIN_JSONL.exists() and VAL_JSONL.exists():
        return
    run(
        [
            sys.executable,
            "training/prepare_hf_reasoning_jsonl.py",
            "--dataset_id",
            DATASET_ID,
            "--tokenizer_name",
            MODEL_NAME,
            "--output_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--val_jsonl",
            path_for_cli(VAL_JSONL),
            "--limit",
            str(OPUS_LIMIT),
            "--val_fraction",
            str(VAL_FRACTION),
            "--max_total_tokens",
            str(MAX_TOTAL_TOKENS),
        ],
        log_name="prepare_opus.log",
    )


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
            path_for_cli(RECOVERED_CHECKPOINT),
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


def phase2_config(projection: Path) -> Path:
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": MAX_LENGTH,
        "max_loops": 4,
        "num_trajectories": NUM_TRAJECTORIES,
        "sample_latents": False,
        "particle_update_mode": "svgd",
        "particle_init_noise": float(PARTICLE_INIT_NOISE),
        "svgd_eps": 1.0,
        "svgd_repulsion_scale": float(REPULSION_SCALE),
        "svgd_bandwidth": "median",
        "svgd_bandwidth_floor": 1e-6,
        "svgd_repulsion_max_norm": None,
        "svgd_kernel_projection_path": path_for_cli(projection),
        "svgd_kernel_projection_dim": int(PROJECTION_DIM),
        "svgd_kernel_geometry": "euclidean",
        "svgd_projection_seed": 0,
        "latent_dim": 256,
        "latent_scale_init": 0.01,
        "latent_adapter_std": 1e-4,
        "latent_injection_mode": "pre",
        "initial_halt_prob": 0.15,
        "beta": BETA,
        "eta": 0.0,
        "rho": 1e-3,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_steps": MAX_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 5,
        "train_on_prompt": False,
        "output_dir": path_for_cli(RUN_DIR / "phase2"),
        "resume_from": path_for_cli(RECOVERED_CHECKPOINT),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
    }
    cfg_path = RUN_DIR / "phase2_smoke.yaml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def latest_phase2_checkpoint() -> Path:
    checkpoints = sorted((RUN_DIR / "phase2").glob("phase2_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No phase2 checkpoints under {RUN_DIR / 'phase2'}")
    return checkpoints[-1]


def train_phase2(cfg_path: Path) -> Path:
    run(
        [
            sys.executable,
            "training/train_phase2_stochastic.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--device",
            DEVICE,
        ],
        log_name="phase2_train.log",
    )
    return latest_phase2_checkpoint()


def eval_phase2(checkpoint: Path, projection: Path) -> Path:
    run_id = f"{RUN_ID}_arc{ARC_LIMIT}"
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
            "STAGE5_BENCHMARK_CHECKPOINT": path_for_cli(checkpoint),
            "STAGE5_BENCHMARKS": "arc_challenge",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": ARC_LIMIT,
            "STAGE5_BENCHMARK_SCORE_TARGETS": "label",
            "STAGE5_BENCHMARK_AGGREGATES": "mean,max,vote",
            "STAGE5_BENCHMARK_RECURRENT_MODE": "phase2",
            "STAGE5_BENCHMARK_NUM_TRAJECTORIES": str(NUM_TRAJECTORIES),
            "STAGE5_BENCHMARK_PARTICLE_UPDATE_MODE": "svgd",
            "STAGE5_BENCHMARK_PARTICLE_INIT_NOISE": PARTICLE_INIT_NOISE,
            "STAGE5_BENCHMARK_SVGD_REPULSION_SCALE": REPULSION_SCALE,
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
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env, log_name="phase2_arc_benchmark.log")
    return ROOT / "outputs" / "stage5" / run_id / "summary.json"


def backup_to_drive() -> None:
    mount_drive_if_possible()
    root = drive_root()
    if not root.exists():
        print(f"Drive backup skipped; missing {root}")
        return
    backup = root / RUN_ID / "run_dir"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup, dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(checkpoint: Path, projection: Path, benchmark_summary: Path) -> None:
    benchmark = read_json(benchmark_summary)
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_recovered_phase2_smoke",
        "parent_checkpoint": path_for_cli(RECOVERED_CHECKPOINT),
        "phase2_checkpoint": path_for_cli(checkpoint),
        "projection": path_for_cli(projection),
        "benchmark_summary": path_for_cli(benchmark_summary),
        "arc_limit": ARC_LIMIT,
        "benchmark": benchmark,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    comparisons = benchmark["comparisons"]["arc_challenge"]["label"]
    paired = benchmark["paired_comparisons"]["arc_challenge"]["label"]
    lines = [
        f"# Recovered Phase2 Smoke - {RUN_ID}",
        "",
        f"- Parent checkpoint: `{path_for_cli(RECOVERED_CHECKPOINT)}`",
        f"- Phase2 checkpoint: `{path_for_cli(checkpoint)}`",
        f"- ARC limit: `{ARC_LIMIT}`",
        f"- K: `{NUM_TRAJECTORIES}`",
        f"- Repulsion: `{REPULSION_SCALE}`",
        "",
        "## ARC-Challenge",
    ]
    for aggregate, row in comparisons.items():
        paired_row = paired[aggregate]
        lines.append(
            f"- `{aggregate}`: recurrent `{row['recurrent']['correct']}/{row['recurrent']['total']}` "
            f"vs base `{row['base']['correct']}/{row['base']['total']}`, delta "
            f"`{row['correct_delta_recurrent_vs_base']}`, W/L/T "
            f"`{paired_row['wins']}/{paired_row['losses']}/{paired_row['ties']}`, p "
            f"`{paired_row['sign_test_p_value']}`"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    for pattern in ["*.yaml", "*.json", "*.md", "*.log"]:
        run(["git", "add", "-f", path_for_cli(RUN_DIR / pattern)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No recovered Phase2 smoke outputs changed.")
        return
    run(["git", "commit", "-m", f"Record recovered Phase2 smoke {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    restore_checkpoint_if_needed(RECOVERED_CHECKPOINT, run_id=RECOVERED_RUN_ID)
    prepare_opus()
    projection = calibrate_projection()
    cfg_path = phase2_config(projection)
    checkpoint = train_phase2(cfg_path)
    benchmark_summary = eval_phase2(checkpoint, projection)
    backup_to_drive()
    write_summary(checkpoint, projection, benchmark_summary)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
