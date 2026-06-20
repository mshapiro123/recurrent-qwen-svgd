"""Gate deterministic ARC recovery before spending more time on particles.

This runner asks two separate questions:

1. Does targeted ARC-style SFT improve the surgically altered recurrent model?
2. After that deterministic recovery, do K-particle/SVGD candidates add value
   over the recovered recurrent baseline?

The point is to avoid crediting particles for gains that came from ordinary
training, or dismissing particles before the recurrent baseline is competent
enough for their alternatives to matter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_RUN_ID") or time.strftime(
    "stage5_arc_agi_recovery_particle_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

SFT_RUN_ID = f"{RUN_ID}_synthetic_sft"
SYNTHETIC_TASKS = int(os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_TASKS", "200"))
SYNTHETIC_SEED = int(os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_SEED", "101"))
SYNTHETIC_MODES = os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_MODES", "all")
TRACE_MODE = os.environ.get("STAGE5_ARC_AGI_RECOVERY_TRACE_MODE", "symbolic_program")
TRACE_FILTER = os.environ.get("STAGE5_ARC_AGI_RECOVERY_TRACE_FILTER", "covered")
TRAIN_STEPS = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_STEPS", "300"))
TRAIN_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "100"))
EVAL_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "20"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "prefer")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")

PARTICLE_TRAJECTORIES = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_TRAJECTORIES", "4"))
PARTICLE_NOISE = float(os.environ.get("STAGE5_ARC_AGI_PARTICLE_NOISE", "0.01"))
PARTICLE_NOISE_STEPS = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_NOISE_STEPS", "16"))
PARTICLE_PROJECTION_DIM = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_PROJECTION_DIM", "32"))
PARTICLE_VARIANTS = os.environ.get(
    "STAGE5_ARC_AGI_PARTICLE_VARIANTS",
    "k4_noise0_rep0:0:0,k4_noise001_rep0:0.01:0,k4_noise001_rep05:0.01:0.5,k4_noise001_rep2:0.01:2",
)


@dataclass(frozen=True)
class ParticleVariant:
    name: str
    noise: float
    repulsion: float


def parse_particle_variants(value: str) -> list[ParticleVariant]:
    variants: list[ParticleVariant] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Particle variants must be comma-separated name:noise:repulsion items. "
                f"Got {item!r}."
            )
        variants.append(ParticleVariant(parts[0], float(parts[1]), float(parts[2])))
    if not variants:
        raise ValueError("At least one particle variant is required.")
    return variants


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], key: str) -> int:
    return int(summary.get(key, 0))


def rate(summary: dict[str, Any], key: str) -> float:
    return float(summary.get(key, 0.0))


def compare_summaries(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_delta": metric(candidate, "selected_exact") - metric(reference, "selected_exact"),
        "best_of_k_delta": metric(candidate, "best_of_k_exact") - metric(reference, "best_of_k_exact"),
        "first_delta": metric(candidate, "first_exact") - metric(reference, "first_exact"),
        "valid_rate_delta": rate(candidate, "valid_candidate_rate") - rate(reference, "valid_candidate_rate"),
    }


def decide_recovery(sft_summary: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    tuned = sft_summary["phase1_arc_agi_tuned"]
    start = sft_summary["phase1_start"]
    base = sft_summary["base"]
    tuned_vs_start = compare_summaries(tuned, start)
    tuned_vs_base = compare_summaries(tuned, base)
    decision = tuned_vs_start["best_of_k_delta"] >= 0 and tuned_vs_start["selected_delta"] >= 0
    evidence = {
        "phase1_tuned_vs_start": tuned_vs_start,
        "phase1_tuned_vs_base": tuned_vs_base,
        "phase1_start": start,
        "phase1_tuned": tuned,
        "base": base,
        "recovery_non_negative": decision,
    }
    return decision, evidence


def decide_particle_value(
    particle_summaries: dict[str, dict[str, Any]],
    tuned_summary: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    rows: dict[str, Any] = {}
    for name, summary in particle_summaries.items():
        rows[name] = {
            "summary": summary,
            "delta_vs_tuned": compare_summaries(summary, tuned_summary),
        }
    best_variant = None
    for name, row in rows.items():
        delta = row["delta_vs_tuned"]
        if delta["best_of_k_delta"] >= 0 and delta["selected_delta"] >= 0:
            if best_variant is None or delta["best_of_k_delta"] > rows[best_variant]["delta_vs_tuned"]["best_of_k_delta"]:
                best_variant = name
    return best_variant is not None, {"variants": rows, "best_non_negative_variant": best_variant}


def run_synthetic_sft() -> dict[str, Any]:
    summary_path = ROOT / "outputs" / "stage5" / SFT_RUN_ID / "summary.json"
    if summary_path.exists() and os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        print(f"reusing_sft_summary={summary_path}")
        return read_json(summary_path)

    env = os.environ.copy()
    env.update(
        {
            "STAGE5_ARC_AGI_SFT_RUN_ID": SFT_RUN_ID,
            "STAGE5_ARC_AGI_SFT_PUSH": "0",
            "STAGE5_ARC_AGI_TRACE_MODE": TRACE_MODE,
            "STAGE5_ARC_AGI_TRACE_FILTER": TRACE_FILTER,
            "STAGE5_ARC_AGI_SYNTHETIC_TASKS": str(SYNTHETIC_TASKS),
            "STAGE5_ARC_AGI_SYNTHETIC_SEED": str(SYNTHETIC_SEED),
            "STAGE5_ARC_AGI_SYNTHETIC_MODES": SYNTHETIC_MODES,
            "STAGE5_ARC_AGI_TRAIN_STEPS": str(TRAIN_STEPS),
            "STAGE5_ARC_AGI_SAVE_EVERY": str(max(TRAIN_STEPS, 1)),
            "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT": str(TRAIN_TASK_LIMIT),
            "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": str(EVAL_TASK_LIMIT),
            "STAGE5_ARC_AGI_MAX_NEW_TOKENS": str(MAX_NEW_TOKENS),
            "STAGE5_ARC_AGI_GRID_FORMAT": GRID_FORMAT,
            "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": PROGRAM_PARSE_MODE,
            "DTYPE": DTYPE,
            "ADAPTER_DTYPE": ADAPTER_DTYPE,
            "DEVICE": DEVICE,
        }
    )
    run([sys.executable, "colab/run_stage5_arc_agi_sft.py"], env=env, log_name="synthetic_sft.log")
    return read_json(summary_path)


def eval_particle_variant(
    *,
    variant: ParticleVariant,
    tasks_path: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{variant.name}_summary.json"
    summary_md = RUN_DIR / f"{variant.name}_summary.md"
    output_jsonl = RUN_DIR / f"{variant.name}_candidates.jsonl"
    if summary_json.exists() and os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        print(f"reusing_particle_summary={summary_json}")
        return read_json(summary_json)["summary"]
    run(
        [
            sys.executable,
            "eval/eval_arc_agi.py",
            "--tasks_path",
            str(tasks_path),
            "--limit",
            str(EVAL_TASK_LIMIT),
            "--mode",
            "phase2",
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_trajectories",
            str(PARTICLE_TRAJECTORIES),
            "--particle_update_mode",
            "svgd",
            "--particle_init_noise",
            str(variant.noise),
            "--particle_noise_every_step",
            "--particle_noise_steps",
            str(PARTICLE_NOISE_STEPS),
            "--svgd_repulsion_scale",
            str(variant.repulsion),
            "--svgd_kernel_projection_dim",
            str(PARTICLE_PROJECTION_DIM),
            "--svgd_kernel_geometry",
            "euclidean",
            "--svgd_repulsion_max_norm",
            "none",
            "--max_new_tokens",
            str(MAX_NEW_TOKENS),
            "--grid_format",
            GRID_FORMAT,
            "--program_parse_mode",
            PROGRAM_PARSE_MODE,
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--output_jsonl",
            path_for_cli(output_jsonl),
            "--summary_json",
            path_for_cli(summary_json),
            "--summary_md",
            path_for_cli(summary_md),
        ],
        log_name=f"{variant.name}.log",
    )
    return read_json(summary_json)["summary"]


def backup_to_drive() -> None:
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Colab only
            print(f"Drive mount skipped/failed: {exc}")
            return
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    child_dir = ROOT / "outputs" / "stage5" / SFT_RUN_ID
    if child_dir.exists():
        shutil.copytree(child_dir, backup / "synthetic_sft", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No recovery-particle gate outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI recovery particle gate"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    recovery = payload["recovery_decision"]
    particle = payload["particle_decision"]
    lines = [
        f"# Stage 5 ARC-AGI Recovery vs Particles - {RUN_ID}",
        "",
        "## Decisions",
        "",
        f"- Deterministic recurrent recovery non-negative: `{recovery['passed']}`",
        f"- Particle/SVGD non-negative over tuned recurrent: `{particle['passed']}`",
        f"- Best non-negative particle variant: `{particle['evidence'].get('best_non_negative_variant')}`",
        "",
        "## Recovery Evidence",
        "",
        f"- Tuned vs start: `{recovery['evidence']['phase1_tuned_vs_start']}`",
        f"- Tuned vs base: `{recovery['evidence']['phase1_tuned_vs_base']}`",
        "",
        "## Particle Evidence",
        "",
    ]
    for name, row in particle["evidence"]["variants"].items():
        lines.append(f"- `{name}` delta vs tuned: `{row['delta_vs_tuned']}` summary `{row['summary']}`")
    lines.extend(
        [
            "",
            "Interpretation: particle variants only count as promising if they are measured against the tuned recurrent baseline, not against the pre-SFT recurrent checkpoint.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab to test deterministic ARC recovery before particle/SVGD value.")
        return 0

    particle_variants = parse_particle_variants(PARTICLE_VARIANTS)
    sft_summary = run_synthetic_sft()
    recovery_passed, recovery_evidence = decide_recovery(sft_summary)

    metadata = sft_summary["metadata"]
    tasks_path = Path(metadata["eval_path"])
    tuned_checkpoint = ROOT / sft_summary["tuned_checkpoint"]
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    if not tuned_checkpoint.exists():
        raise FileNotFoundError(tuned_checkpoint)

    particle_summaries = {
        variant.name: eval_particle_variant(variant=variant, tasks_path=tasks_path, checkpoint=tuned_checkpoint)
        for variant in particle_variants
    }
    particle_passed, particle_evidence = decide_particle_value(
        particle_summaries,
        sft_summary["phase1_arc_agi_tuned"],
    )

    payload = {
        "run_id": RUN_ID,
        "synthetic_sft_run_id": SFT_RUN_ID,
        "settings": {
            "synthetic_tasks": SYNTHETIC_TASKS,
            "synthetic_seed": SYNTHETIC_SEED,
            "synthetic_modes": SYNTHETIC_MODES,
            "trace_mode": TRACE_MODE,
            "trace_filter": TRACE_FILTER,
            "train_steps": TRAIN_STEPS,
            "train_task_limit": TRAIN_TASK_LIMIT,
            "eval_task_limit": EVAL_TASK_LIMIT,
            "particle_trajectories": PARTICLE_TRAJECTORIES,
            "particle_noise_steps": PARTICLE_NOISE_STEPS,
            "particle_projection_dim": PARTICLE_PROJECTION_DIM,
            "program_parse_mode": PROGRAM_PARSE_MODE,
            "particle_variants": [variant.__dict__ for variant in particle_variants],
        },
        "sft_summary": sft_summary,
        "recovery_decision": {"passed": recovery_passed, "evidence": recovery_evidence},
        "particle_decision": {"passed": particle_passed, "evidence": particle_evidence},
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
