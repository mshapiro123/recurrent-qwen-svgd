"""Evaluate particles/SVGD on the checkpoint selected by a curriculum run.

Use this after ``run_stage5_arc_agi_curriculum.py`` finishes. It reads the
curriculum summary, recovers the final selected recurrent checkpoint and its
family-level baseline metrics, then runs replicated K-particle inference without
spending GPU time on another SFT pass.
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
RUN_ID = os.environ.get("STAGE5_ARC_AGI_POST_CURRICULUM_PARTICLE_RUN_ID") or time.strftime(
    "stage5_arc_agi_post_curriculum_particle_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_POST_CURRICULUM_PARTICLE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

CURRICULUM_SUMMARY = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_SUMMARY", "")
TASKS_PATH_OVERRIDE = os.environ.get("STAGE5_ARC_AGI_POST_CURRICULUM_TASKS_PATH", "")
EVAL_TASK_LIMIT_OVERRIDE = os.environ.get("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "")
PROGRAM_PARSE_MODE_OVERRIDE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "")
SELECTION_STRATEGY_OVERRIDE = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "")
GRID_FORMAT_OVERRIDE = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "")
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")

PARTICLE_TRAJECTORIES = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_TRAJECTORIES", "4"))
PARTICLE_SEEDS = os.environ.get("STAGE5_ARC_AGI_PARTICLE_SEEDS", "0,1,2")
PARTICLE_NOISE_STEPS = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_NOISE_STEPS", "16"))
PARTICLE_PROJECTION_DIM = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_PROJECTION_DIM", "32"))
PARTICLE_VARIANTS = os.environ.get(
    "STAGE5_ARC_AGI_PARTICLE_VARIANTS",
    "k4_noise0_rep0:0:0,k4_noise001_rep0:0.01:0,k4_noise001_rep05:0.01:0.5,k4_noise001_rep2:0.01:2",
)

from colab.run_stage5_arc_agi_recovery_particle_gate import (  # noqa: E402
    decide_seeded_particle_value,
    parse_int_csv,
    parse_particle_variants,
    task_family_summary_from_payload,
)


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


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_curriculum_summary() -> Path:
    candidates = sorted(
        ROOT.glob("outputs/stage5/stage5_arc*_curriculum_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            ROOT.glob("outputs/stage5/*curriculum*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError("No curriculum summary found. Set STAGE5_ARC_AGI_CURRICULUM_SUMMARY.")
    return candidates[0]


def resolve_curriculum_summary() -> Path:
    if CURRICULUM_SUMMARY:
        return resolve_path(CURRICULUM_SUMMARY)
    return latest_curriculum_summary()


def final_stage_row(curriculum: dict[str, Any]) -> dict[str, Any]:
    stages = curriculum.get("stages") or []
    if not stages:
        raise ValueError("Curriculum summary has no stages.")
    return stages[-1]


def child_summary_for_stage(stage: dict[str, Any]) -> dict[str, Any]:
    child_dir = resolve_path(stage["child_run_dir"])
    return read_json(child_dir / "summary.json")


def curriculum_context(curriculum_summary_path: str | Path) -> dict[str, Any]:
    summary_path = resolve_path(curriculum_summary_path)
    curriculum = read_json(summary_path)
    final_stage = final_stage_row(curriculum)
    child_summary = child_summary_for_stage(final_stage)
    selected = final_stage["selected_checkpoint"]
    checkpoint = resolve_path(selected["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    metadata = child_summary.get("metadata", {})
    tasks_path = resolve_path(TASKS_PATH_OVERRIDE or metadata["eval_path"])
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    eval_limit = int(EVAL_TASK_LIMIT_OVERRIDE or metadata.get("eval_task_limit", 20))
    program_parse_mode = PROGRAM_PARSE_MODE_OVERRIDE or metadata.get("program_parse_mode", "prefer")
    selection_strategy = SELECTION_STRATEGY_OVERRIDE or metadata.get("selection_strategy", "heuristic")
    grid_format = GRID_FORMAT_OVERRIDE or metadata.get("grid_format", "compact")
    return {
        "curriculum_summary_path": path_for_cli(summary_path),
        "curriculum": curriculum,
        "final_stage": final_stage,
        "child_summary": child_summary,
        "checkpoint": checkpoint,
        "reference_summary": selected["summary"],
        "reference_task_family_summary": task_family_summary_from_payload(selected),
        "tasks_path": tasks_path,
        "eval_limit": eval_limit,
        "program_parse_mode": program_parse_mode,
        "selection_strategy": selection_strategy,
        "grid_format": grid_format,
    }


def eval_particle_variant(
    *,
    variant_name: str,
    noise: float,
    repulsion: float,
    seed: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    label = f"{variant_name}_seed{seed}"
    summary_json = RUN_DIR / f"{label}_summary.json"
    output_jsonl = RUN_DIR / f"{label}_candidates.jsonl"
    summary_md = RUN_DIR / f"{label}_summary.md"
    if summary_json.exists() and os.environ.get("STAGE5_ARC_AGI_POST_CURRICULUM_PARTICLE_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        payload = read_json(summary_json)
        payload["seed"] = seed
        return payload

    run(
        [
            sys.executable,
            "eval/eval_arc_agi.py",
            "--tasks_path",
            str(context["tasks_path"]),
            "--limit",
            str(context["eval_limit"]),
            "--mode",
            "phase2",
            "--checkpoint",
            path_for_cli(context["checkpoint"]),
            "--seed",
            str(seed),
            "--max_loops",
            "4",
            "--num_trajectories",
            str(PARTICLE_TRAJECTORIES),
            "--particle_update_mode",
            "svgd",
            "--particle_init_noise",
            str(noise),
            "--particle_noise_every_step",
            "--particle_noise_steps",
            str(PARTICLE_NOISE_STEPS),
            "--svgd_repulsion_scale",
            str(repulsion),
            "--svgd_kernel_projection_dim",
            str(PARTICLE_PROJECTION_DIM),
            "--svgd_kernel_geometry",
            "euclidean",
            "--svgd_repulsion_max_norm",
            "none",
            "--max_new_tokens",
            str(MAX_NEW_TOKENS),
            "--grid_format",
            str(context["grid_format"]),
            "--program_parse_mode",
            str(context["program_parse_mode"]),
            "--selection_strategy",
            str(context["selection_strategy"]),
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
        log_name=f"{label}.log",
    )
    payload = read_json(summary_json)
    payload["seed"] = seed
    return payload


def eval_seeded_particles(context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seeds = parse_int_csv(PARTICLE_SEEDS)
    variants = parse_particle_variants(PARTICLE_VARIANTS)
    return {
        variant.name: [
            eval_particle_variant(
                variant_name=variant.name,
                noise=variant.noise,
                repulsion=variant.repulsion,
                seed=seed,
                context=context,
            )
            for seed in seeds
        ]
        for variant in variants
    }


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
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No post-curriculum particle outputs changed.")
        return
    run(["git", "commit", "-m", "Record post-curriculum ARC particle gate"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    decision = payload["particle_decision"]
    lines = [
        f"# Stage 5 Post-Curriculum Particle Gate - {RUN_ID}",
        "",
        f"- Curriculum summary: `{payload['context']['curriculum_summary_path']}`",
        f"- Checkpoint: `{payload['context']['checkpoint']}`",
        f"- Tasks path: `{payload['context']['tasks_path']}`",
        f"- Eval limit: `{payload['context']['eval_limit']}`",
        f"- Program parse mode: `{payload['context']['program_parse_mode']}`",
        f"- Selection strategy: `{payload['context']['selection_strategy']}`",
        f"- Particle decision passed: `{decision['passed']}`",
        f"- Best replicated variant: `{decision['evidence'].get('best_replicated_variant')}`",
        "",
        "## Reference Recurrent Summary",
        "",
        f"- Summary: `{payload['reference_summary']}`",
        f"- Task families: `{payload['reference_task_family_summary']}`",
        "",
        "## Particle Variants",
        "",
    ]
    for name, row in decision["evidence"]["variants"].items():
        lines.append(
            f"- `{name}` mean delta `{row['mean_delta_vs_tuned']}` "
            f"non_negative `{row['non_negative_seed_count']}` / `{row['evaluated_seed_count']}` "
            f"passed `{row['passed']}`"
        )
        lines.append(f"  - family mean deltas: `{row['task_family_mean_delta_vs_tuned']}`")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run after a curriculum run. Set STAGE5_ARC_AGI_CURRICULUM_SUMMARY or use latest.")
        return 0

    summary_path = resolve_curriculum_summary()
    context = curriculum_context(summary_path)
    seeded_particles = eval_seeded_particles(context)
    passed, evidence = decide_seeded_particle_value(
        seeded_particles,
        context["reference_summary"],
        context["reference_task_family_summary"],
    )
    payload = {
        "run_id": RUN_ID,
        "settings": {
            "particle_variants": [variant.__dict__ for variant in parse_particle_variants(PARTICLE_VARIANTS)],
            "particle_seeds": parse_int_csv(PARTICLE_SEEDS),
            "particle_trajectories": PARTICLE_TRAJECTORIES,
            "particle_noise_steps": PARTICLE_NOISE_STEPS,
            "particle_projection_dim": PARTICLE_PROJECTION_DIM,
        },
        "context": {
            "curriculum_summary_path": context["curriculum_summary_path"],
            "checkpoint": path_for_cli(context["checkpoint"]),
            "tasks_path": str(context["tasks_path"]),
            "eval_limit": context["eval_limit"],
            "program_parse_mode": context["program_parse_mode"],
            "selection_strategy": context["selection_strategy"],
            "grid_format": context["grid_format"],
            "final_stage": context["final_stage"],
        },
        "reference_summary": context["reference_summary"],
        "reference_task_family_summary": context["reference_task_family_summary"],
        "seeded_particle_summaries": seeded_particles,
        "particle_decision": {"passed": passed, "evidence": evidence},
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
