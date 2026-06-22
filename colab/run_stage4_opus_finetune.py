"""Run a guarded modified-Opus fine-tuning pass in one Colab runtime.

This runner separates three questions:

1. Does the deterministic recurrent model improve over its prior recurrent baseline?
2. Does the stochastic/SVGD recurrent model improve over deterministic recurrent?
3. How far is the best recurrent model from unmodified base Qwen 0.5B?

It writes checkpoints under ``outputs/stage4/<run_id>`` and commits only configs,
logs, JSONL benchmark outputs, and summaries. Checkpoints are copied to Google
Drive when ``/content/drive/MyDrive`` is mounted.
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

APPROVED_STAGE4_OPUS_DATASETS = {
    "lordx64/reasoning-distill-opus-4-7-max-sft",
    "lordx64/reasoning-distill-claude-opus-4-7-max",
}

RUN_ID = os.environ.get("STAGE4_RUN_ID") or time.strftime("stage4_opus_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage4" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_ID = os.environ.get("OPUS_DATASET_ID", "lordx64/reasoning-distill-opus-4-7-max-sft")
DATASET_NAME = os.environ.get("OPUS_DATASET_NAME", "")
DATASET_ADAPTER = os.environ.get("OPUS_DATASET_ADAPTER", "auto")
ALLOW_UNAPPROVED_DATASET = os.environ.get("STAGE4_OPUS_ALLOW_UNAPPROVED_SOURCE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_NO_DRIVE_BACKUP = os.environ.get("STAGE4_ALLOW_NO_DRIVE_BACKUP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DATASET_LIMIT = int(os.environ.get("OPUS_LIMIT", "3000"))
VAL_FRACTION = float(os.environ.get("OPUS_VAL_FRACTION", "0.05"))
MAX_TOTAL_TOKENS = int(os.environ.get("OPUS_MAX_TOTAL_TOKENS", "1024"))
MAX_LENGTH = int(os.environ.get("STAGE4_MAX_LENGTH", "512"))
PHASE1_STEPS = int(os.environ.get("STAGE4_PHASE1_STEPS", "500"))
PHASE1_SAVE_EVERY = int(os.environ.get("STAGE4_PHASE1_SAVE_EVERY", "250"))
PHASE2_STEPS = int(os.environ.get("STAGE4_PHASE2_STEPS", "100"))
PHASE2_SAVE_EVERY = int(os.environ.get("STAGE4_PHASE2_SAVE_EVERY", "50"))
ARC_LIMIT = int(os.environ.get("STAGE4_ARC_LIMIT", "128"))
EXACT_SEEDS = os.environ.get("STAGE4_EXACT_SEEDS", "0,1,2")
RUN_EXACT = os.environ.get("STAGE4_RUN_EXACT", "0").strip().lower() in {"1", "true", "yes", "y"}
EXACT_MAX_NEW_TOKENS = int(os.environ.get("STAGE4_EXACT_MAX_NEW_TOKENS", "96"))
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")

TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_opus_train.jsonl"
VAL_JSONL = ROOT / "data" / f"{RUN_ID}_opus_val.jsonl"
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
    returncode = process.wait()
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def validate_dataset_source(
    *,
    dataset_id: str = DATASET_ID,
    allow_unapproved: bool = ALLOW_UNAPPROVED_DATASET,
) -> dict[str, Any]:
    approved = dataset_id in APPROVED_STAGE4_OPUS_DATASETS
    payload = {
        "dataset_id": dataset_id,
        "approved": approved,
        "allow_unapproved": allow_unapproved,
        "approved_datasets": sorted(APPROVED_STAGE4_OPUS_DATASETS),
    }
    if approved or allow_unapproved:
        return payload
    raise ValueError(
        "Stage 4 Opus fine-tune is restricted to approved recovery datasets "
        f"{sorted(APPROVED_STAGE4_OPUS_DATASETS)}. Got {dataset_id!r}. "
        "Set STAGE4_OPUS_ALLOW_UNAPPROVED_SOURCE=1 only for a deliberate, "
        "non-default experiment."
    )


def drive_backup_root() -> Path:
    return Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))


def validate_drive_backup(
    *,
    drive_root: Path | None = None,
    allow_no_backup: bool = ALLOW_NO_DRIVE_BACKUP,
) -> dict[str, Any]:
    root = drive_root or drive_backup_root()
    payload = {
        "drive_root": str(root),
        "available": root.exists(),
        "allow_no_backup": allow_no_backup,
    }
    if root.exists() or allow_no_backup:
        return payload
    raise RuntimeError(
        f"Stage 4 fine-tune requires a mounted Drive backup directory before A100 training: {root}. "
        "Mount/authorize Google Drive first, or set STAGE4_ALLOW_NO_DRIVE_BACKUP=1 for a deliberate "
        "non-default run."
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_mcq(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_agg: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_agg.setdefault(str(row["aggregate"]), []).append(row)
    return {
        agg: {
            "correct": sum(1 for row in agg_rows if row["hit"]),
            "total": len(agg_rows),
            "accuracy": sum(1 for row in agg_rows if row["hit"]) / max(len(agg_rows), 1),
        }
        for agg, agg_rows in sorted(by_agg.items())
    }


def accuracy(summary: dict[str, Any], label: str = "mean") -> float | None:
    metric = summary.get(label)
    if not metric:
        return None
    return float(metric["accuracy"])


def subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def summarize_ladder(
    base: dict[str, Any],
    phase1: dict[str, Any],
    phase2: dict[str, Any],
) -> dict[str, Any]:
    base_mean = accuracy(base, "mean")
    phase1_mean = accuracy(phase1, "mean")
    phase2_mean = accuracy(phase2, "mean")
    phase2_max = accuracy(phase2, "max")
    phase2_vote = accuracy(phase2, "vote")
    best_phase2 = max(
        [value for value in [phase2_mean, phase2_max, phase2_vote] if value is not None],
        default=None,
    )
    return {
        "base_mean_accuracy": base_mean,
        "phase1_mean_accuracy": phase1_mean,
        "phase2_mean_accuracy": phase2_mean,
        "phase2_max_accuracy": phase2_max,
        "phase2_vote_accuracy": phase2_vote,
        "phase2_best_accuracy": best_phase2,
        "phase1_gap_to_base": subtract_optional(phase1_mean, base_mean),
        "phase2_mean_lift_over_phase1": subtract_optional(phase2_mean, phase1_mean),
        "phase2_best_lift_over_phase1": subtract_optional(best_phase2, phase1_mean),
        "phase2_best_gap_to_base": subtract_optional(best_phase2, base_mean),
    }


def summarize_jsonl_eval(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def eval_jsonl(label: str, checkpoint: Path, *, phase2: bool = False) -> dict[str, float]:
    cmd = [
        sys.executable,
        "eval/eval_jsonl.py",
        "--model_name",
        MODEL_NAME,
        "--data_jsonl",
        str(VAL_JSONL.relative_to(ROOT)),
        "--checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--split",
        "6,18",
        "--max_loops",
        "4",
        "--max_length",
        str(MAX_LENGTH),
        "--beta",
        "0.08",
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
    ]
    if phase2:
        cmd += [
            "--num_trajectories",
            "4",
            "--particle_update_mode",
            "svgd",
            "--particle_init_noise",
            "0.02",
            "--svgd_repulsion_scale",
            "0.5",
            "--svgd_repulsion_max_norm",
            "1.0",
            "--rho",
            "0.001",
        ]
    proc = run(cmd, log_name=f"{label}_val.log")
    return summarize_jsonl_eval(proc.stdout)


def eval_arc(label: str, mode: str, checkpoint: Path | None = None, projection: Path | None = None) -> Path:
    output = RUN_DIR / f"arc_{label}.jsonl"
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--data_jsonl",
        str(ARC_JSONL.relative_to(ROOT)),
        "--prompt_style",
        "with_options",
        "--score_target",
        "label",
        "--mode",
        mode,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--seed",
        "0",
    ]
    if mode == "base":
        cmd += ["--aggregate", "mean"]
    elif mode == "phase1":
        assert checkpoint is not None
        cmd += [
            "--checkpoint",
            str(checkpoint.relative_to(ROOT)),
            "--max_loops",
            "4",
            "--num_trajectories",
            "1",
            "--aggregate",
            "mean",
        ]
    else:
        assert checkpoint is not None and projection is not None
        cmd += [
            "--checkpoint",
            str(checkpoint.relative_to(ROOT)),
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
            str(projection.relative_to(ROOT)),
            "--svgd_kernel_projection_dim",
            "8",
            "--svgd_repulsion_scale",
            "2",
            "--svgd_repulsion_max_norm",
            "none",
        ]
    cmd += ["--output_jsonl", str(output.relative_to(ROOT))]
    run(cmd, log_name=f"arc_{label}.log")
    return output


def backup_to_drive() -> None:
    drive_root = drive_backup_root()
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    if RUN_DIR.exists():
        shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for source in [TRAIN_JSONL, VAL_JSONL, ARC_JSONL]:
        if source.exists():
            target = backup / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"backed_up_to={backup}")


def ensure_drive_mount() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    print("Drive is not mounted; Stage 4 will stop unless STAGE4_ALLOW_NO_DRIVE_BACKUP=1 is set.")


def main() -> int:
    ensure_drive_mount()
    drive_backup_preflight = validate_drive_backup()
    dataset_source_preflight = validate_dataset_source()
    metadata = {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME or None,
        "dataset_adapter": DATASET_ADAPTER,
        "dataset_source_preflight": dataset_source_preflight,
        "drive_backup_preflight": drive_backup_preflight,
        "dataset_limit": DATASET_LIMIT,
        "val_fraction": VAL_FRACTION,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "max_length": MAX_LENGTH,
        "phase1_steps": PHASE1_STEPS,
        "phase2_steps": PHASE2_STEPS,
        "arc_limit": ARC_LIMIT,
        "exact_seeds": EXACT_SEEDS,
        "run_exact": RUN_EXACT,
        "exact_max_new_tokens": EXACT_MAX_NEW_TOKENS,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    if not TRAIN_JSONL.exists() or not VAL_JSONL.exists():
        prepare_cmd = [
            sys.executable,
            "training/prepare_hf_reasoning_jsonl.py",
            "--dataset_id",
            DATASET_ID,
            "--adapter",
            DATASET_ADAPTER,
            "--tokenizer_name",
            MODEL_NAME,
            "--output_jsonl",
            str(TRAIN_JSONL.relative_to(ROOT)),
            "--val_jsonl",
            str(VAL_JSONL.relative_to(ROOT)),
            "--limit",
            str(DATASET_LIMIT),
            "--val_fraction",
            str(VAL_FRACTION),
            "--max_total_tokens",
            str(MAX_TOTAL_TOKENS),
        ]
        if DATASET_NAME:
            prepare_cmd += ["--name", DATASET_NAME]
        run(prepare_cmd, log_name="prepare_opus.log")

    phase1_cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": MAX_LENGTH,
        "max_loops": 4,
        "initial_halt_prob": 0.15,
        "beta": 0.08,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": PHASE1_STEPS,
        "save_every": PHASE1_SAVE_EVERY,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": str((RUN_DIR / "phase1").relative_to(ROOT)),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
    }
    phase1_cfg_path = RUN_DIR / "phase1.yaml"
    write_yaml(phase1_cfg_path, phase1_cfg)
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            str(phase1_cfg_path.relative_to(ROOT)),
            "--train_jsonl",
            str(TRAIN_JSONL.relative_to(ROOT)),
            "--device",
            DEVICE,
        ],
        log_name="phase1_train.log",
    )
    phase1_ckpt = RUN_DIR / "phase1" / f"phase1_step_{PHASE1_STEPS}.pt"
    assert phase1_ckpt.exists(), phase1_ckpt
    phase1_val = eval_jsonl("phase1", phase1_ckpt)

    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            "ARC-Challenge",
            "--split",
            "validation",
            "--limit",
            str(ARC_LIMIT),
            "--seed",
            "0",
            "--output_jsonl",
            str(ARC_JSONL.relative_to(ROOT)),
        ],
        log_name="prepare_arc.log",
    )
    arc_base = eval_arc("base_label", "base")
    arc_phase1 = eval_arc("phase1_label", "phase1", phase1_ckpt)

    phase2_cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": MAX_LENGTH,
        "max_loops": 4,
        "num_trajectories": 4,
        "sample_latents": False,
        "particle_update_mode": "svgd",
        "particle_init_noise": 0.02,
        "svgd_eps": 1.0,
        "svgd_repulsion_scale": 0.5,
        "svgd_bandwidth": "median",
        "svgd_bandwidth_floor": 1e-6,
        "svgd_repulsion_max_norm": 1.0,
        "latent_dim": 256,
        "latent_scale_init": 0.01,
        "latent_adapter_std": 1e-4,
        "latent_injection_mode": "pre",
        "initial_halt_prob": 0.15,
        "beta": 0.08,
        "eta": 0.0,
        "rho": 0.001,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": PHASE2_STEPS,
        "save_every": PHASE2_SAVE_EVERY,
        "log_every": 10,
        "train_on_prompt": False,
        "output_dir": str((RUN_DIR / "phase2").relative_to(ROOT)),
        "resume_from": str(phase1_ckpt.relative_to(ROOT)),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
    }
    phase2_cfg_path = RUN_DIR / "phase2.yaml"
    write_yaml(phase2_cfg_path, phase2_cfg)
    run(
        [
            sys.executable,
            "training/train_phase2_stochastic.py",
            "--config",
            str(phase2_cfg_path.relative_to(ROOT)),
            "--train_jsonl",
            str(TRAIN_JSONL.relative_to(ROOT)),
            "--device",
            DEVICE,
        ],
        log_name="phase2_train.log",
    )
    phase2_ckpt = RUN_DIR / "phase2" / f"phase2_step_{PHASE2_STEPS}.pt"
    assert phase2_ckpt.exists(), phase2_ckpt
    phase2_val = eval_jsonl("phase2", phase2_ckpt, phase2=True)

    projection = RUN_DIR / "within_group_projection.pt"
    run(
        [
            sys.executable,
            "eval/calibrate_svgd_projection.py",
            "--tasks_jsonl",
            "eval/smoke_exact_tasks_v2.jsonl",
            "--phase2_checkpoint",
            str(phase2_ckpt.relative_to(ROOT)),
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
            str(projection.relative_to(ROOT)),
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
        ],
        log_name="calibrate_projection.log",
    )
    arc_phase2 = eval_arc("phase2_svgd_label", "phase2", phase2_ckpt, projection)

    exact_out = RUN_DIR / "exact_phase1_vs_phase2.jsonl"
    if RUN_EXACT:
        run(
            [
                sys.executable,
                "eval/eval_best_of_k_jsonl.py",
                "--tasks_jsonl",
                "eval/smoke_exact_tasks_v2.jsonl",
                "--compact",
                "--seeds",
                EXACT_SEEDS,
                "--phase1_checkpoint",
                str(phase1_ckpt.relative_to(ROOT)),
                "--phase2_checkpoint",
                str(phase2_ckpt.relative_to(ROOT)),
                "--phase2_num_trajectories",
                "4",
                "--phase2_particle_update_mode",
                "svgd",
                "--particle_init_noise",
                "0.05",
                "--particle_noise_every_step",
                "--particle_noise_steps",
                "16",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                str(projection.relative_to(ROOT)),
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "2",
                "--svgd_repulsion_max_norm",
                "none",
                "--temperature",
                "0.0",
                "--max_new_tokens",
                str(EXACT_MAX_NEW_TOKENS),
                "--dtype",
                DTYPE,
                "--adapter_dtype",
                ADAPTER_DTYPE,
                "--device",
                DEVICE,
                "--output_jsonl",
                str(exact_out.relative_to(ROOT)),
            ],
            log_name="exact_phase1_vs_phase2.log",
        )
    else:
        (RUN_DIR / "exact_phase1_vs_phase2.log").write_text(
            "Skipped. Set STAGE4_RUN_EXACT=1 to run exact generation diagnostics.\n",
            encoding="utf-8",
        )

    arc_summary = {
        "base": summarize_mcq(arc_base),
        "phase1_recurrent_baseline": summarize_mcq(arc_phase1),
        "phase2_recurrent_candidate": summarize_mcq(arc_phase2),
    }
    ladder = summarize_ladder(
        arc_summary["base"],
        arc_summary["phase1_recurrent_baseline"],
        arc_summary["phase2_recurrent_candidate"],
    )
    summary = {
        "run_id": RUN_ID,
        "kind": "stage4_opus_finetune",
        "checkpoint": str(phase1_ckpt.relative_to(ROOT)),
        "checkpoint_role": "phase1_deterministic_recurrent",
        "phase1_checkpoint": str(phase1_ckpt.relative_to(ROOT)),
        "phase2_checkpoint": str(phase2_ckpt.relative_to(ROOT)),
        "projection": str(projection.relative_to(ROOT)),
        "phase1_val": phase1_val,
        "phase2_val": phase2_val,
        "arc": arc_summary,
        "arc_ladder": ladder,
        "exact_generation_ran": RUN_EXACT,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 4 Opus Fine-Tune Summary - {RUN_ID}",
        "",
        "## Framing",
        "Base Qwen is the outer target. Phase1 is the deterministic recurrent baseline. Phase2/SVGD is the recurrent candidate.",
        "",
        "## Validation",
        f"- Phase1 val: `{phase1_val}`",
        f"- Phase2 val: `{phase2_val}`",
        "",
        "## ARC Label-Likelihood",
    ]
    for label, metrics in summary["arc"].items():
        lines.append(f"- {label}: `{metrics}`")
    lines.extend(
        [
            "",
            "## Baseline Ladder",
            f"- Base mean accuracy: `{ladder['base_mean_accuracy']}`",
            f"- Phase1 recurrent mean accuracy: `{ladder['phase1_mean_accuracy']}`",
            f"- Phase2 recurrent best accuracy: `{ladder['phase2_best_accuracy']}`",
            f"- Phase1 gap to base: `{ladder['phase1_gap_to_base']}`",
            f"- Phase2 best lift over Phase1: `{ladder['phase2_best_lift_over_phase1']}`",
            f"- Phase2 best gap to base: `{ladder['phase2_best_gap_to_base']}`",
            "",
            "Interpretation rule: Phase2 is a useful recurrent mechanism only if it first beats Phase1. Beating base Qwen is the later training target.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))

    backup_to_drive()

    run(["git", "status", "-sb"])
    run(["git", "add", "-f", str((RUN_DIR / "*.yaml").relative_to(ROOT))], check=False)
    run(["git", "add", "-f", str((RUN_DIR / "*.json").relative_to(ROOT))], check=False)
    run(["git", "add", "-f", str((RUN_DIR / "*.md").relative_to(ROOT))], check=False)
    run(["git", "add", "-f", str((RUN_DIR / "*.log").relative_to(ROOT))], check=False)
    run(["git", "add", "-f", str((RUN_DIR / "arc_*.jsonl").relative_to(ROOT))], check=False)
    run(["git", "add", "-f", str((RUN_DIR / "exact_phase1_vs_phase2.jsonl").relative_to(ROOT))], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No stage4 summary changes to commit.")
    else:
        run(["git", "commit", "-m", "Record Stage 4 Opus fine-tune run"])
        run(["git", "push", "origin", "main"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
