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
SYNTHETIC_EVAL_TASKS = int(os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_EVAL_TASKS", "20"))
SYNTHETIC_EVAL_SEED = int(os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_EVAL_SEED", str(SYNTHETIC_SEED + 100_000)))
SYNTHETIC_EVAL_MODES = os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_EVAL_MODES", SYNTHETIC_MODES)
SYNTHETIC_EVAL_PARSE_MODES = os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_EVAL_PARSE_MODES", "")
TRACE_MODE = os.environ.get("STAGE5_ARC_AGI_RECOVERY_TRACE_MODE", "symbolic_program")
TRACE_FILTER = os.environ.get("STAGE5_ARC_AGI_RECOVERY_TRACE_FILTER", "covered")
TRAIN_STEPS = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_STEPS", "300"))
SAVE_EVERY = int(os.environ.get("STAGE5_ARC_AGI_SAVE_EVERY", str(max(TRAIN_STEPS, 1))))
EVAL_CHECKPOINT_LADDER = os.environ.get("STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
TRAIN_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "100"))
EVAL_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "20"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "prefer")
SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "heuristic")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")

PARTICLE_TRAJECTORIES = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_TRAJECTORIES", "4"))
PARTICLE_NOISE = float(os.environ.get("STAGE5_ARC_AGI_PARTICLE_NOISE", "0.01"))
PARTICLE_NOISE_STEPS = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_NOISE_STEPS", "16"))
PARTICLE_PROJECTION_DIM = int(os.environ.get("STAGE5_ARC_AGI_PARTICLE_PROJECTION_DIM", "32"))
PARTICLE_CHECKPOINT_LADDER = os.environ.get("STAGE5_ARC_AGI_PARTICLE_CHECKPOINT_LADDER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PARTICLE_SEEDS = os.environ.get("STAGE5_ARC_AGI_PARTICLE_SEEDS", "0")
PARTICLE_VARIANTS = os.environ.get(
    "STAGE5_ARC_AGI_PARTICLE_VARIANTS",
    "k4_noise0_rep0:0:0,k4_noise001_rep0:0.01:0,k4_noise001_rep05:0.01:0.5,k4_noise001_rep2:0.01:2",
)
SYNTHETIC_HOLDOUT_JSON = RUN_DIR / "synthetic_holdout_tasks.json"


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def summary_from_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Accept either a raw summary or a full eval payload."""

    summary = value.get("summary")
    return summary if isinstance(summary, dict) else value


def task_family_summary_from_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Accept either a full eval payload or compact Colab eval diagnostics."""

    direct = value.get("task_family_summary")
    if isinstance(direct, dict):
        return direct
    diagnostics = value.get("eval_diagnostics")
    if isinstance(diagnostics, dict):
        nested = diagnostics.get("task_family_summary")
        if isinstance(nested, dict):
            return nested
    return {}


def compare_task_family_summaries(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for family in sorted(set(candidate) | set(reference)):
        cand = candidate.get(family, {})
        ref = reference.get(family, {})
        rows[family] = compare_summaries(cand, ref)
    return rows


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    seeds = [int(item) for item in parse_csv(value)]
    return seeds or [0]


def synthetic_eval_parse_modes() -> list[str]:
    modes = parse_csv(SYNTHETIC_EVAL_PARSE_MODES)
    return modes or [PROGRAM_PARSE_MODE]


def select_recovered_checkpoint(sft_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the checkpoint/summary that should represent recovered Phase1."""

    best = sft_summary.get("best_checkpoint") or {}
    if best.get("checkpoint") and best.get("summary"):
        return {
            "source": "best_checkpoint",
            "checkpoint": best["checkpoint"],
            "summary": best["summary"],
            "step": best.get("step"),
        }
    return {
        "source": "final_checkpoint",
        "checkpoint": sft_summary.get("tuned_checkpoint"),
        "summary": sft_summary["phase1_arc_agi_tuned"],
        "step": None,
    }


def recovered_task_family_summary(sft_summary: dict[str, Any], recovered: dict[str, Any]) -> dict[str, Any]:
    if recovered.get("source") == "best_checkpoint":
        best = sft_summary.get("best_checkpoint") or {}
        best_family = task_family_summary_from_payload(best)
        if best_family:
            return best_family
    tuned_diagnostics = (sft_summary.get("eval_diagnostics") or {}).get("phase1_arc_agi_tuned", {})
    return task_family_summary_from_payload(tuned_diagnostics)


def decide_recovery(sft_summary: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    recovered = select_recovered_checkpoint(sft_summary)
    tuned = recovered["summary"]
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
        "phase1_recovered": recovered,
        "base": base,
        "recovery_non_negative": decision,
    }
    return decision, evidence


def decide_particle_value(
    particle_summaries: dict[str, dict[str, Any]],
    tuned_summary: dict[str, Any],
    tuned_task_family_summary: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    rows: dict[str, Any] = {}
    tuned_task_family_summary = tuned_task_family_summary or {}
    for name, payload_or_summary in particle_summaries.items():
        summary = summary_from_payload(payload_or_summary)
        task_family_summary = task_family_summary_from_payload(payload_or_summary)
        rows[name] = {
            "summary": summary,
            "task_family_summary": task_family_summary,
            "delta_vs_tuned": compare_summaries(summary, tuned_summary),
            "task_family_delta_vs_tuned": compare_task_family_summaries(
                task_family_summary,
                tuned_task_family_summary,
            ),
        }
    best_variant = None
    for name, row in rows.items():
        delta = row["delta_vs_tuned"]
        if delta["best_of_k_delta"] >= 0 and delta["selected_delta"] >= 0:
            if best_variant is None or delta["best_of_k_delta"] > rows[best_variant]["delta_vs_tuned"]["best_of_k_delta"]:
                best_variant = name
    return best_variant is not None, {"variants": rows, "best_non_negative_variant": best_variant}


def mean_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("selected_delta", "best_of_k_delta", "first_delta", "valid_rate_delta")
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)
        for key in keys
    }


def decide_seeded_particle_value(
    seeded_particle_summaries: dict[str, list[dict[str, Any]]],
    tuned_summary: dict[str, Any],
    tuned_task_family_summary: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    tuned_task_family_summary = tuned_task_family_summary or {}
    variants: dict[str, Any] = {}
    best_variant = None
    for name, payloads in seeded_particle_summaries.items():
        seed_rows: list[dict[str, Any]] = []
        family_delta_rows: dict[str, list[dict[str, Any]]] = {}
        for payload in payloads:
            summary = summary_from_payload(payload)
            delta = compare_summaries(summary, tuned_summary)
            family_delta = compare_task_family_summaries(
                task_family_summary_from_payload(payload),
                tuned_task_family_summary,
            )
            for family, row in family_delta.items():
                family_delta_rows.setdefault(family, []).append(row)
            seed_rows.append(
                {
                    "seed": payload.get("seed"),
                    "summary": summary,
                    "delta_vs_tuned": delta,
                    "task_family_delta_vs_tuned": family_delta,
                    "non_negative": delta["best_of_k_delta"] >= 0 and delta["selected_delta"] >= 0,
                }
            )

        mean = mean_delta([row["delta_vs_tuned"] for row in seed_rows])
        non_negative_count = sum(1 for row in seed_rows if row["non_negative"])
        family_mean_delta = {
            family: mean_delta(rows)
            for family, rows in sorted(family_delta_rows.items())
        }
        passed = (
            non_negative_count > len(seed_rows) / 2
            and mean["best_of_k_delta"] >= 0
            and mean["selected_delta"] >= 0
        )
        variants[name] = {
            "seeds": seed_rows,
            "mean_delta_vs_tuned": mean,
            "task_family_mean_delta_vs_tuned": family_mean_delta,
            "non_negative_seed_count": non_negative_count,
            "evaluated_seed_count": len(seed_rows),
            "passed": passed,
        }
        if passed and (
            best_variant is None
            or mean["best_of_k_delta"] > variants[best_variant]["mean_delta_vs_tuned"]["best_of_k_delta"]
            or (
                mean["best_of_k_delta"] == variants[best_variant]["mean_delta_vs_tuned"]["best_of_k_delta"]
                and mean["selected_delta"] > variants[best_variant]["mean_delta_vs_tuned"]["selected_delta"]
            )
        ):
            best_variant = name
    return best_variant is not None, {"variants": variants, "best_replicated_variant": best_variant}


def summarize_particle_ladder(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether particles help at particular recovery checkpoints."""

    by_variant: dict[str, Any] = {}
    best_row: dict[str, Any] | None = None
    for row in rows:
        variant = str(row["variant"])
        delta = row["delta_vs_recurrent"]
        is_non_negative = delta["best_of_k_delta"] >= 0 and delta["selected_delta"] >= 0
        variant_row = by_variant.setdefault(
            variant,
            {
                "evaluated_checkpoints": 0,
                "non_negative_checkpoints": 0,
                "best_selected_delta": None,
                "best_best_of_k_delta": None,
                "best_step": None,
            },
        )
        variant_row["evaluated_checkpoints"] += 1
        if is_non_negative:
            variant_row["non_negative_checkpoints"] += 1
        if (
            best_row is None
            or delta["best_of_k_delta"] > best_row["delta_vs_recurrent"]["best_of_k_delta"]
            or (
                delta["best_of_k_delta"] == best_row["delta_vs_recurrent"]["best_of_k_delta"]
                and delta["selected_delta"] > best_row["delta_vs_recurrent"]["selected_delta"]
            )
        ):
            best_row = row
        if (
            variant_row["best_best_of_k_delta"] is None
            or delta["best_of_k_delta"] > variant_row["best_best_of_k_delta"]
            or (
                delta["best_of_k_delta"] == variant_row["best_best_of_k_delta"]
                and (
                    variant_row["best_selected_delta"] is None
                    or delta["selected_delta"] > variant_row["best_selected_delta"]
                )
            )
        ):
            variant_row["best_best_of_k_delta"] = delta["best_of_k_delta"]
            variant_row["best_selected_delta"] = delta["selected_delta"]
            variant_row["best_step"] = row.get("step")

    return {
        "evaluated_rows": len(rows),
        "by_variant": by_variant,
        "best_row": best_row,
    }


def summarize_holdout_recovery(holdout: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_parse_mode: dict[str, Any] = {}
    for parse_mode, rows in holdout.items():
        tuned = rows["phase1_tuned"]["summary"]
        start = rows["phase1_start"]["summary"]
        base = rows["base"]["summary"]
        by_parse_mode[parse_mode] = {
            "phase1_tuned_vs_start": compare_summaries(tuned, start),
            "phase1_tuned_vs_base": compare_summaries(tuned, base),
            "base": base,
            "phase1_start": start,
            "phase1_tuned": tuned,
            "parse_methods": {
                name: row.get("parse_method_summary", {})
                for name, row in rows.items()
            },
        }
    return by_parse_mode


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
            "STAGE5_ARC_AGI_SAVE_EVERY": str(SAVE_EVERY),
            "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER": "1" if EVAL_CHECKPOINT_LADDER else "0",
            "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT": str(TRAIN_TASK_LIMIT),
            "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": str(EVAL_TASK_LIMIT),
            "STAGE5_ARC_AGI_MAX_NEW_TOKENS": str(MAX_NEW_TOKENS),
            "STAGE5_ARC_AGI_GRID_FORMAT": GRID_FORMAT,
            "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": PROGRAM_PARSE_MODE,
            "STAGE5_ARC_AGI_SELECTION_STRATEGY": SELECTION_STRATEGY,
            "DTYPE": DTYPE,
            "ADAPTER_DTYPE": ADAPTER_DTYPE,
            "DEVICE": DEVICE,
        }
    )
    run([sys.executable, "colab/run_stage5_arc_agi_sft.py"], env=env, log_name="synthetic_sft.log")
    return read_json(summary_path)


def generate_synthetic_holdout() -> Path | None:
    if SYNTHETIC_EVAL_TASKS <= 0:
        return None
    if SYNTHETIC_HOLDOUT_JSON.exists() and os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        print(f"reusing_synthetic_holdout={SYNTHETIC_HOLDOUT_JSON}")
        return SYNTHETIC_HOLDOUT_JSON
    run(
        [
            sys.executable,
            "training/generate_arc_agi_synthetic_tasks.py",
            "--output_json",
            path_for_cli(SYNTHETIC_HOLDOUT_JSON),
            "--num_tasks",
            str(SYNTHETIC_EVAL_TASKS),
            "--seed",
            str(SYNTHETIC_EVAL_SEED),
            "--modes",
            SYNTHETIC_EVAL_MODES,
        ],
        log_name="generate_synthetic_holdout.log",
    )
    return SYNTHETIC_HOLDOUT_JSON


def eval_arc_variant(
    *,
    label: str,
    mode: str,
    tasks_path: Path,
    checkpoint: Path | None,
    program_parse_mode: str,
) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{label}_summary.json"
    summary_md = RUN_DIR / f"{label}_summary.md"
    output_jsonl = RUN_DIR / f"{label}_candidates.jsonl"
    if summary_json.exists() and os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        print(f"reusing_eval_summary={summary_json}")
        return read_json(summary_json)

    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(tasks_path),
        "--limit",
        str(SYNTHETIC_EVAL_TASKS if tasks_path == SYNTHETIC_HOLDOUT_JSON else EVAL_TASK_LIMIT),
        "--mode",
        mode,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--program_parse_mode",
        program_parse_mode,
        "--selection_strategy",
        SELECTION_STRATEGY,
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
    ]
    if mode != "base":
        if checkpoint is None:
            raise ValueError(f"checkpoint is required for mode={mode}")
        cmd += [
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_candidates",
            "1",
        ]
    run(cmd, log_name=f"{label}.log")
    return read_json(summary_json)


def run_synthetic_holdout_evals(
    *,
    sft_summary: dict[str, Any],
    tuned_checkpoint: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    holdout_path = generate_synthetic_holdout()
    if holdout_path is None:
        return {}
    phase1_start = resolve_path(sft_summary["metadata"]["phase1_checkpoint"])
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for parse_mode in synthetic_eval_parse_modes():
        results[parse_mode] = {
            "base": eval_arc_variant(
                label=f"synthetic_holdout_{parse_mode}_base",
                mode="base",
                tasks_path=holdout_path,
                checkpoint=None,
                program_parse_mode=parse_mode,
            ),
            "phase1_start": eval_arc_variant(
                label=f"synthetic_holdout_{parse_mode}_phase1_start",
                mode="phase1",
                tasks_path=holdout_path,
                checkpoint=phase1_start,
                program_parse_mode=parse_mode,
            ),
            "phase1_tuned": eval_arc_variant(
                label=f"synthetic_holdout_{parse_mode}_phase1_tuned",
                mode="phase1",
                tasks_path=holdout_path,
                checkpoint=tuned_checkpoint,
                program_parse_mode=parse_mode,
            ),
        }
    return results


def eval_particle_variant(
    *,
    variant: ParticleVariant,
    tasks_path: Path,
    checkpoint: Path,
    seed: int = 0,
    label: str | None = None,
) -> dict[str, Any]:
    label = label or f"{variant.name}_seed{seed}"
    summary_json = RUN_DIR / f"{label}_summary.json"
    summary_md = RUN_DIR / f"{label}_summary.md"
    output_jsonl = RUN_DIR / f"{label}_candidates.jsonl"
    if summary_json.exists() and os.environ.get("STAGE5_ARC_AGI_RECOVERY_PARTICLE_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        print(f"reusing_particle_summary={summary_json}")
        payload = read_json(summary_json)
        payload["seed"] = seed
        return payload
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
            "--seed",
            str(seed),
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
            "--selection_strategy",
            SELECTION_STRATEGY,
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


def eval_seeded_particle_variants(
    *,
    particle_variants: list[ParticleVariant],
    tasks_path: Path,
    checkpoint: Path,
    seeds: list[int],
) -> dict[str, list[dict[str, Any]]]:
    return {
        variant.name: [
            eval_particle_variant(
                variant=variant,
                tasks_path=tasks_path,
                checkpoint=checkpoint,
                seed=seed,
                label=f"{variant.name}_seed{seed}",
            )
            for seed in seeds
        ]
        for variant in particle_variants
    }


def eval_particle_checkpoint_ladder(
    *,
    sft_summary: dict[str, Any],
    tasks_path: Path,
    particle_variants: list[ParticleVariant],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_row in sft_summary.get("checkpoint_ladder") or []:
        checkpoint_value = checkpoint_row.get("checkpoint")
        if not checkpoint_value:
            continue
        checkpoint = resolve_path(checkpoint_value)
        if not checkpoint.exists():
            print(f"skipping_missing_ladder_checkpoint={checkpoint}")
            continue
        step = checkpoint_row.get("step")
        recurrent_summary = checkpoint_row["summary"]
        for variant in particle_variants:
            label = f"ladder_step{step}_{variant.name}"
            particle_summary = eval_particle_variant(
                variant=variant,
                tasks_path=tasks_path,
                checkpoint=checkpoint,
                seed=0,
                label=label,
            )
            particle_summary_metrics = summary_from_payload(particle_summary)
            rows.append(
                {
                    "step": step,
                    "checkpoint": path_for_cli(checkpoint),
                    "variant": variant.name,
                    "recurrent_summary": recurrent_summary,
                    "recurrent_task_family_summary": task_family_summary_from_payload(checkpoint_row),
                    "particle_summary": particle_summary_metrics,
                    "particle_task_family_summary": task_family_summary_from_payload(particle_summary),
                    "delta_vs_recurrent": compare_summaries(particle_summary_metrics, recurrent_summary),
                    "task_family_delta_vs_recurrent": compare_task_family_summaries(
                        task_family_summary_from_payload(particle_summary),
                        task_family_summary_from_payload(checkpoint_row),
                    ),
                }
            )
    return rows


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
        f"- Particle/SVGD non-negative over recovered recurrent: `{particle['passed']}`",
        f"- Best replicated particle variant: `{particle['evidence'].get('best_replicated_variant')}`",
        f"- First-seed particle decision: `{payload.get('first_seed_particle_decision', {}).get('passed')}`",
        "",
        "## Recovery Evidence",
        "",
        f"- Recovered checkpoint: `{recovery['evidence']['phase1_recovered']}`",
        f"- Recovered vs start: `{recovery['evidence']['phase1_tuned_vs_start']}`",
        f"- Recovered vs base: `{recovery['evidence']['phase1_tuned_vs_base']}`",
        "",
    ]
    holdout = recovery["evidence"].get("synthetic_holdout", {})
    if holdout:
        lines.extend(["## Synthetic Holdout Evidence", ""])
        for parse_mode, row in holdout.items():
            lines.extend(
                [
                    f"### Parse mode `{parse_mode}`",
                    "",
                    f"- Recovered vs start: `{row['phase1_tuned_vs_start']}`",
                    f"- Recovered vs base: `{row['phase1_tuned_vs_base']}`",
                    f"- Parse methods: `{row['parse_methods']}`",
                    "",
                ]
            )
    lines.extend(["## Particle Evidence", ""])
    for name, row in particle["evidence"]["variants"].items():
        if "mean_delta_vs_tuned" in row:
            lines.append(
                f"- `{name}` mean delta vs tuned: `{row['mean_delta_vs_tuned']}` "
                f"non_negative_seeds `{row['non_negative_seed_count']}` / `{row['evaluated_seed_count']}` "
                f"passed `{row['passed']}`"
            )
            family_delta = row.get("task_family_mean_delta_vs_tuned", {})
        else:
            lines.append(f"- `{name}` delta vs tuned: `{row['delta_vs_tuned']}` summary `{row['summary']}`")
            family_delta = row.get("task_family_delta_vs_tuned", {})
        if family_delta:
            lines.append(f"  - task family deltas: `{family_delta}`")
    particle_ladder = payload.get("particle_checkpoint_ladder_summary", {})
    if particle_ladder:
        lines.extend(
            [
                "",
                "## Particle Checkpoint Ladder",
                "",
                f"- Evaluated rows: `{particle_ladder['evaluated_rows']}`",
                f"- By variant: `{particle_ladder['by_variant']}`",
                f"- Best row: `{particle_ladder['best_row']}`",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: particle variants only count as promising if they are measured against the selected recovered recurrent checkpoint, not against the pre-SFT recurrent checkpoint or an arbitrary final SFT endpoint.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab to test deterministic ARC recovery before particle/SVGD value.")
        return 0

    particle_variants = parse_particle_variants(PARTICLE_VARIANTS)
    particle_seeds = parse_int_csv(PARTICLE_SEEDS)
    sft_summary = run_synthetic_sft()
    recovery_passed, recovery_evidence = decide_recovery(sft_summary)
    recovered = select_recovered_checkpoint(sft_summary)
    recovered_family_summary = recovered_task_family_summary(sft_summary, recovered)

    metadata = sft_summary["metadata"]
    tasks_path = resolve_path(metadata["eval_path"])
    if not recovered.get("checkpoint"):
        raise FileNotFoundError("Missing recovered checkpoint path in SFT summary.")
    tuned_checkpoint = resolve_path(recovered["checkpoint"])
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    if not tuned_checkpoint.exists():
        raise FileNotFoundError(tuned_checkpoint)

    synthetic_holdout = run_synthetic_holdout_evals(
        sft_summary=sft_summary,
        tuned_checkpoint=tuned_checkpoint,
    )
    if synthetic_holdout:
        recovery_evidence["synthetic_holdout"] = summarize_holdout_recovery(synthetic_holdout)

    seeded_particle_summaries = eval_seeded_particle_variants(
        particle_variants=particle_variants,
        tasks_path=tasks_path,
        checkpoint=tuned_checkpoint,
        seeds=particle_seeds,
    )
    first_seed_particle_summaries = {
        name: payloads[0]
        for name, payloads in seeded_particle_summaries.items()
        if payloads
    }
    first_seed_particle_passed, first_seed_particle_evidence = decide_particle_value(
        first_seed_particle_summaries,
        recovered["summary"],
        tuned_task_family_summary=recovered_family_summary,
    )
    particle_passed, particle_evidence = decide_seeded_particle_value(
        seeded_particle_summaries,
        recovered["summary"],
        tuned_task_family_summary=recovered_family_summary,
    )
    particle_ladder_rows = (
        eval_particle_checkpoint_ladder(
            sft_summary=sft_summary,
            tasks_path=tasks_path,
            particle_variants=particle_variants,
        )
        if PARTICLE_CHECKPOINT_LADDER
        else []
    )
    particle_ladder_summary = summarize_particle_ladder(particle_ladder_rows) if particle_ladder_rows else {}

    payload = {
        "run_id": RUN_ID,
        "synthetic_sft_run_id": SFT_RUN_ID,
        "settings": {
            "synthetic_tasks": SYNTHETIC_TASKS,
            "synthetic_seed": SYNTHETIC_SEED,
            "synthetic_modes": SYNTHETIC_MODES,
            "synthetic_eval_tasks": SYNTHETIC_EVAL_TASKS,
            "synthetic_eval_seed": SYNTHETIC_EVAL_SEED,
            "synthetic_eval_modes": SYNTHETIC_EVAL_MODES,
            "synthetic_eval_parse_modes": synthetic_eval_parse_modes(),
            "trace_mode": TRACE_MODE,
            "trace_filter": TRACE_FILTER,
            "train_steps": TRAIN_STEPS,
            "save_every": SAVE_EVERY,
            "eval_checkpoint_ladder": EVAL_CHECKPOINT_LADDER,
            "train_task_limit": TRAIN_TASK_LIMIT,
            "eval_task_limit": EVAL_TASK_LIMIT,
            "particle_trajectories": PARTICLE_TRAJECTORIES,
            "particle_seeds": particle_seeds,
            "particle_noise_steps": PARTICLE_NOISE_STEPS,
            "particle_projection_dim": PARTICLE_PROJECTION_DIM,
            "particle_checkpoint_ladder": PARTICLE_CHECKPOINT_LADDER,
            "program_parse_mode": PROGRAM_PARSE_MODE,
            "selection_strategy": SELECTION_STRATEGY,
            "particle_variants": [variant.__dict__ for variant in particle_variants],
        },
        "recovered_checkpoint": recovered,
        "recovered_task_family_summary": recovered_family_summary,
        "sft_summary": sft_summary,
        "synthetic_holdout": synthetic_holdout,
        "recovery_decision": {"passed": recovery_passed, "evidence": recovery_evidence},
        "particle_decision": {"passed": particle_passed, "evidence": particle_evidence},
        "first_seed_particle_decision": {
            "passed": first_seed_particle_passed,
            "evidence": first_seed_particle_evidence,
        },
        "seeded_particle_summaries": seeded_particle_summaries,
        "particle_checkpoint_ladder": particle_ladder_rows,
        "particle_checkpoint_ladder_summary": particle_ladder_summary,
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
