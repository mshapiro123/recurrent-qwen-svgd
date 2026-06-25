"""Run a broader Stage 5 benchmark suite for a recurrent adapter artifact.

This is the first post-release-gate benchmark runner. It compares unmodified
base Qwen against the selected recurrent checkpoint on MCQ-style reasoning
slices, starting with ARC-Challenge, ARC-Easy, and GPQA-lite. Prepared question
data stays under ``data/`` and is not committed; result JSONLs omit
question/choice text and are written under ``outputs/stage5``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.colab_auth import ensure_hf_token_from_colab
from colab.run_stage5_publish_hf_adapter import checkpoint_value_from_payload
from colab.run_stage5_recovered_phase1_arc_gate import (
    candidate_drive_checkpoints,
    drive_diagnostics,
    mount_drive_if_possible,
)
from eval.analyze_mcq_regressions import (
    paired_rows as paired_mcq_regression_rows,
    rows_by_id as mcq_rows_by_id,
    summarize as summarize_mcq_regressions,
)
from eval.mcq_debias import aggregate_permutation_scores, cyclic_permutation_rows, read_jsonl, write_jsonl


RUN_ID = os.environ.get("STAGE5_BENCHMARK_SUITE_RUN_ID") or time.strftime(
    "stage5_benchmark_suite_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DATA_DIR = ROOT / "data" / "stage5_benchmark_suite" / RUN_ID

def suite_profile_defaults(profile: str) -> dict[str, str]:
    """Return environment defaults for named benchmark profiles.

    Profiles keep Colab launch cells small and make the scientific intent of a
    run explicit. The depth confirmation profile deliberately uses an open
    ARC-Challenge hard slice as the third leg so the run can complete without
    gated GPQA access.
    """

    normalized = profile.strip().lower()
    if normalized in {"", "default", "debiased"}:
        return {
            "benchmarks": "arc_easy,arc_challenge,gpqa_lite",
            "score_targets": "label,content_question_only,cyclic_label_aggregated",
            "arc_challenge_limit": "128",
            "arc_easy_limit": "128",
            "open_hard_arc_challenge_limit": "256",
        }
    if normalized in {"depth_signal_confirmation", "depth_signal"}:
        return {
            "benchmarks": "arc_easy,arc_challenge,open_hard_arc_challenge",
            "score_targets": "content_question_only,cyclic_label_aggregated",
            "arc_challenge_limit": "256",
            "arc_easy_limit": "128",
            "open_hard_arc_challenge_limit": "256",
        }
    raise ValueError(f"Unknown STAGE5_BENCHMARK_SUITE_PROFILE={profile!r}")


SUITE_PROFILE = os.environ.get("STAGE5_BENCHMARK_SUITE_PROFILE", "default")
PROFILE_DEFAULTS = suite_profile_defaults(SUITE_PROFILE)
SOURCE_SUMMARY = os.environ.get("STAGE5_BENCHMARK_SOURCE_SUMMARY", "")
EXPLICIT_CHECKPOINT = os.environ.get("STAGE5_BENCHMARK_CHECKPOINT", "")
BENCHMARKS = os.environ.get("STAGE5_BENCHMARKS", PROFILE_DEFAULTS["benchmarks"])
ARC_CHALLENGE_LIMIT_RAW = os.environ.get("STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT", PROFILE_DEFAULTS["arc_challenge_limit"])
ARC_EASY_LIMIT_RAW = os.environ.get("STAGE5_BENCHMARK_ARC_EASY_LIMIT", PROFILE_DEFAULTS["arc_easy_limit"])
ARC_CHALLENGE_OFFSET = int(os.environ.get("STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET", "0"))
ARC_EASY_OFFSET = int(os.environ.get("STAGE5_BENCHMARK_ARC_EASY_OFFSET", "0"))
OPEN_HARD_ARC_CHALLENGE_LIMIT_RAW = os.environ.get(
    "STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_LIMIT",
    PROFILE_DEFAULTS["open_hard_arc_challenge_limit"],
)
OPEN_HARD_ARC_CHALLENGE_OFFSET = int(os.environ.get("STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_OFFSET", "0"))
GPQA_LIMIT = int(os.environ.get("STAGE5_BENCHMARK_GPQA_LIMIT", "16"))
GPQA_CONFIG = os.environ.get("STAGE5_BENCHMARK_GPQA_CONFIG", "gpqa_diamond")
SCORE_TARGETS = os.environ.get("STAGE5_BENCHMARK_SCORE_TARGETS", PROFILE_DEFAULTS["score_targets"])
AGGREGATES = os.environ.get("STAGE5_BENCHMARK_AGGREGATES", "mean")
CONTINUE_ON_FAILURE = os.environ.get("STAGE5_BENCHMARK_CONTINUE_ON_FAILURE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

RECURRENT_MODE = os.environ.get("STAGE5_BENCHMARK_RECURRENT_MODE", "phase1")
RECURRENT_MAX_LOOPS = int(os.environ.get("STAGE5_BENCHMARK_MAX_LOOPS", "4"))
RECURRENT_NUM_TRAJECTORIES = int(os.environ.get("STAGE5_BENCHMARK_NUM_TRAJECTORIES", "1"))
RECURRENT_SAMPLE_LATENTS = os.environ.get("STAGE5_BENCHMARK_SAMPLE_LATENTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RECURRENT_LATENT_INJECTION_MODE = os.environ.get("STAGE5_BENCHMARK_LATENT_INJECTION_MODE", "post")
RECURRENT_PARTICLE_UPDATE_MODE = os.environ.get("STAGE5_BENCHMARK_PARTICLE_UPDATE_MODE", "none")
RECURRENT_PARTICLE_INIT_NOISE = os.environ.get("STAGE5_BENCHMARK_PARTICLE_INIT_NOISE", "0.0")
RECURRENT_SVGD_EPS = os.environ.get("STAGE5_BENCHMARK_SVGD_EPS", "1.0")
RECURRENT_SVGD_REPULSION_SCALE = os.environ.get("STAGE5_BENCHMARK_SVGD_REPULSION_SCALE", "0.5")
RECURRENT_SVGD_BANDWIDTH = os.environ.get("STAGE5_BENCHMARK_SVGD_BANDWIDTH", "median")
RECURRENT_SVGD_BANDWIDTH_FLOOR = os.environ.get("STAGE5_BENCHMARK_SVGD_BANDWIDTH_FLOOR", "1e-6")
RECURRENT_SVGD_REPULSION_MAX_NORM = os.environ.get("STAGE5_BENCHMARK_SVGD_REPULSION_MAX_NORM", "")
RECURRENT_SVGD_KERNEL_PROJECTION_DIM = os.environ.get("STAGE5_BENCHMARK_SVGD_KERNEL_PROJECTION_DIM", "")
RECURRENT_SVGD_KERNEL_PROJECTION_PATH = os.environ.get("STAGE5_BENCHMARK_SVGD_KERNEL_PROJECTION_PATH", "")
RECURRENT_SVGD_KERNEL_GEOMETRY = os.environ.get("STAGE5_BENCHMARK_SVGD_KERNEL_GEOMETRY", "euclidean")
RECURRENT_SVGD_PROJECTION_SEED = os.environ.get("STAGE5_BENCHMARK_SVGD_PROJECTION_SEED", "0")
RECURRENT_USE_LEARNED_LOOP_CONTROL = os.environ.get(
    "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
INCLUDE_LOOP_DIAGNOSTICS = os.environ.get("STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_BENCHMARK_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}
AFTER_CONFIRM_DENSE_RUN_SUFFIX = os.environ.get("STAGE5_BENCHMARK_AFTER_CONFIRM_DENSE_RUN_SUFFIX", "").strip()
AFTER_CONFIRM_DENSE_EXTRA_TRAIN_JSONL = os.environ.get(
    "STAGE5_BENCHMARK_AFTER_CONFIRM_DENSE_EXTRA_TRAIN_JSONL",
    "",
).strip()


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_jsonl: Path
    prepare_cmd: list[str]


@dataclass(frozen=True)
class EvalJob:
    benchmark: str
    arm: str
    score_target: str
    output_jsonl: Path
    cmd: list[str]
    eval_output_jsonl: Path | None = None
    permutation_jsonl: Path | None = None


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


CYCLIC_LABEL_TARGETS = {"cyclic_label", "cyclic_label_aggregated"}


def public_score_target(score_target: str) -> str:
    if score_target in CYCLIC_LABEL_TARGETS:
        return "cyclic_label_aggregated"
    return score_target


def eval_score_target(score_target: str) -> str:
    if score_target == "content_question_only":
        return "option_text"
    if score_target in CYCLIC_LABEL_TARGETS:
        return "label"
    return score_target


def prompt_style_for_score_target(score_target: str) -> str:
    if score_target == "content_question_only":
        return "question_only"
    return "with_options"


def cyclic_permutation_jsonl(spec: BenchmarkSpec) -> Path:
    path = PRIVATE_DATA_DIR / f"{spec.data_jsonl.stem}_cyclic_permuted.jsonl"
    if not spec.data_jsonl.exists():
        return path
    if not path.exists():
        write_jsonl(path, cyclic_permutation_rows(read_jsonl(spec.data_jsonl)))
    return path


def aggregate_cyclic_label_output(job: EvalJob) -> None:
    if not job.permutation_jsonl or not job.eval_output_jsonl:
        return
    scored_rows = read_jsonl(job.eval_output_jsonl)
    permutation_rows = read_jsonl(job.permutation_jsonl)
    aggregates = sorted({str(row.get("aggregate") or "mean") for row in scored_rows})
    output_rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        aggregate_rows = [row for row in scored_rows if str(row.get("aggregate") or "mean") == aggregate]
        for row in aggregate_permutation_scores(aggregate_rows, permutation_rows):
            row["aggregate"] = f"permutation_{aggregate}"
            output_rows.append(row)
    write_jsonl(job.output_jsonl, output_rows)


def effective_score_targets() -> list[str]:
    targets: list[str] = []
    for score_target in parse_csv(SCORE_TARGETS):
        public = public_score_target(score_target)
        if public not in targets:
            targets.append(public)
    return targets


def configured_score_targets() -> list[str]:
    targets: list[str] = []
    seen_public: set[str] = set()
    for score_target in parse_csv(SCORE_TARGETS):
        public = public_score_target(score_target)
        if public in seen_public:
            continue
        seen_public.add(public)
        targets.append(score_target)
    return targets


def parse_optional_limit(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "full", "all"}:
        return None
    limit = int(normalized)
    if limit <= 0:
        return None
    return limit


ARC_CHALLENGE_LIMIT = parse_optional_limit(ARC_CHALLENGE_LIMIT_RAW)
ARC_EASY_LIMIT = parse_optional_limit(ARC_EASY_LIMIT_RAW)
OPEN_HARD_ARC_CHALLENGE_LIMIT = parse_optional_limit(OPEN_HARD_ARC_CHALLENGE_LIMIT_RAW)


def latest_summary_with_checkpoint() -> Path | None:
    candidates: list[Path] = []
    for root in (ROOT / "outputs" / "hf_exports", ROOT / "outputs" / "stage5"):
        if not root.exists():
            continue
        for path in root.glob("**/summary.json"):
            try:
                payload = read_json(path)
            except Exception:
                continue
            if checkpoint_value_from_payload(payload) or payload.get("checkpoint"):
                candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def resolve_source_summary() -> Path | None:
    if SOURCE_SUMMARY:
        path = resolve_path(SOURCE_SUMMARY)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    return latest_summary_with_checkpoint()


def checkpoint_candidates_from_payload(source_summary: Path | None, payload: dict[str, Any] | None) -> list[Path]:
    candidates: list[Path] = []
    if EXPLICIT_CHECKPOINT:
        candidates.append(resolve_path(EXPLICIT_CHECKPOINT))
    if payload:
        value = checkpoint_value_from_payload(payload) or payload.get("checkpoint")
        if value:
            candidates.append(resolve_path(str(value)))
        export_dir = payload.get("export_dir")
        if export_dir:
            candidates.append(resolve_path(str(export_dir)) / "recurrent_adapter_checkpoint.pt")
    if source_summary:
        candidates.append(source_summary.parent / "recurrent_adapter_checkpoint.pt")
    return candidates


def checkpoint_bearing_source_summary(source_summary: Path | None, payload: dict[str, Any] | None) -> Path | None:
    """Follow no-weight policy/debias wrapper summaries back to the adapter source."""

    if source_summary is None or payload is None:
        return source_summary
    current_summary = source_summary
    current_payload = payload
    seen: set[Path] = set()
    for _depth in range(8):
        if (
            checkpoint_value_from_payload(current_payload)
            or current_payload.get("checkpoint")
            or current_payload.get("export_dir")
            or (current_summary.parent / "recurrent_adapter_checkpoint.pt").exists()
        ):
            return current_summary
        if current_summary in seen:
            raise RuntimeError(f"Cycle while resolving checkpoint-bearing source summary: {current_summary}")
        seen.add(current_summary)
        kind = str(current_payload.get("kind") or "")
        next_summary: str | None = None
        if kind == "stage5_mcq_scoring_policy":
            next_summary = current_payload.get("source_summary")
        elif kind == "stage5_mcq_debias_pair_assessment":
            source_summaries = current_payload.get("source_summaries") or {}
            if isinstance(source_summaries, dict):
                next_summary = source_summaries.get("arc_challenge") or source_summaries.get("arc_easy")
        elif kind == "stage5_mcq_debias_diagnostic":
            next_summary = current_payload.get("nested_source_summary") or current_payload.get("source_summary")
        if not next_summary:
            return current_summary
        current_summary = resolve_path(next_summary)
        current_payload = read_json(current_summary)
    raise RuntimeError(f"Could not resolve checkpoint-bearing source summary from {source_summary}")


def infer_artifact_run_id(path: str | Path) -> str | None:
    parts = Path(path).parts
    for marker in ("stage5", "stage4"):
        for idx, part in enumerate(parts):
            if part == marker and idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def restore_checkpoint_from_drive(candidate: Path) -> Path | None:
    run_id = infer_artifact_run_id(candidate)
    if not run_id:
        return None
    mount_drive_if_possible()
    for drive_candidate in candidate_drive_checkpoints(run_id, candidate.name):
        if drive_candidate.exists():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_candidate, candidate)
            print(f"restored_benchmark_checkpoint={drive_candidate} -> {candidate}", flush=True)
            return candidate
    return None


def resolve_checkpoint(source_summary: Path | None, payload: dict[str, Any] | None) -> Path:
    candidates = checkpoint_candidates_from_payload(source_summary, payload)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in candidates:
        restored = restore_checkpoint_from_drive(candidate)
        if restored and restored.exists():
            return restored
    searched = [path_for_cli(path) for path in candidates]
    raise FileNotFoundError(f"No recurrent checkpoint found. Searched: {searched}\n{drive_diagnostics()}")


def benchmark_specs(names: list[str]) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    for name in names:
        if name in {"arc_challenge", "arc_easy", "open_hard_arc_challenge"}:
            config = "ARC-Challenge" if name == "arc_challenge" else "ARC-Easy"
            limit = ARC_CHALLENGE_LIMIT if name == "arc_challenge" else ARC_EASY_LIMIT
            offset = ARC_CHALLENGE_OFFSET if name == "arc_challenge" else ARC_EASY_OFFSET
            if name == "open_hard_arc_challenge":
                config = "ARC-Challenge"
                limit = OPEN_HARD_ARC_CHALLENGE_LIMIT
                offset = OPEN_HARD_ARC_CHALLENGE_OFFSET
            limit_label = "full" if limit is None else str(limit)
            slice_label = f"offset{offset}_{limit_label}" if offset else limit_label
            prepare_cmd = [
                sys.executable,
                "eval/prepare_arc_mcq.py",
                "--config",
                config,
                "--split",
                "validation",
                "--seed",
                "0",
            ]
            if offset:
                prepare_cmd.extend(["--offset", str(offset)])
            if limit is not None:
                prepare_cmd.extend(["--limit", str(limit)])
            output = PRIVATE_DATA_DIR / f"arc_challenge_validation_{slice_label}.jsonl"
            if name == "arc_easy":
                output = PRIVATE_DATA_DIR / f"arc_easy_validation_{slice_label}.jsonl"
            elif name == "open_hard_arc_challenge":
                output = PRIVATE_DATA_DIR / f"open_hard_arc_challenge_validation_{slice_label}.jsonl"
            prepare_cmd.extend(["--output_jsonl", str(output)])
            specs.append(
                BenchmarkSpec(
                    name=name,
                    data_jsonl=output,
                    prepare_cmd=prepare_cmd,
                )
            )
        elif name == "gpqa_lite":
            output = PRIVATE_DATA_DIR / f"{GPQA_CONFIG}_{GPQA_LIMIT}.jsonl"
            specs.append(
                BenchmarkSpec(
                    name=name,
                    data_jsonl=output,
                    prepare_cmd=[
                        sys.executable,
                        "eval/prepare_gpqa_mcq.py",
                        "--config",
                        GPQA_CONFIG,
                        "--split",
                        "train",
                        "--limit",
                        str(GPQA_LIMIT),
                        "--seed",
                        "0",
                        "--output_jsonl",
                        str(output),
                    ],
                )
            )
        else:
            raise ValueError(
                f"Unknown benchmark {name!r}; expected arc_challenge, arc_easy, open_hard_arc_challenge, or gpqa_lite"
            )
    return specs


def eval_jobs(specs: list[BenchmarkSpec], *, checkpoint: Path) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    recurrent_extra = [
        "--checkpoint",
        path_for_cli(checkpoint),
        "--max_loops",
        str(RECURRENT_MAX_LOOPS),
        "--num_trajectories",
        str(RECURRENT_NUM_TRAJECTORIES),
    ]
    if RECURRENT_USE_LEARNED_LOOP_CONTROL:
        recurrent_extra.append("--use_learned_loop_control")
    if RECURRENT_MODE == "phase2":
        if RECURRENT_SAMPLE_LATENTS:
            recurrent_extra.append("--sample_latents")
        recurrent_extra.extend(
            [
                "--latent_injection_mode",
                RECURRENT_LATENT_INJECTION_MODE,
                "--particle_update_mode",
                RECURRENT_PARTICLE_UPDATE_MODE,
                "--particle_init_noise",
                RECURRENT_PARTICLE_INIT_NOISE,
                "--svgd_eps",
                RECURRENT_SVGD_EPS,
                "--svgd_repulsion_scale",
                RECURRENT_SVGD_REPULSION_SCALE,
                "--svgd_bandwidth",
                RECURRENT_SVGD_BANDWIDTH,
                "--svgd_bandwidth_floor",
                RECURRENT_SVGD_BANDWIDTH_FLOOR,
                "--svgd_kernel_geometry",
                RECURRENT_SVGD_KERNEL_GEOMETRY,
                "--svgd_projection_seed",
                RECURRENT_SVGD_PROJECTION_SEED,
            ]
        )
        if RECURRENT_SVGD_REPULSION_MAX_NORM:
            recurrent_extra.extend(["--svgd_repulsion_max_norm", RECURRENT_SVGD_REPULSION_MAX_NORM])
        if RECURRENT_SVGD_KERNEL_PROJECTION_DIM:
            recurrent_extra.extend(["--svgd_kernel_projection_dim", RECURRENT_SVGD_KERNEL_PROJECTION_DIM])
        if RECURRENT_SVGD_KERNEL_PROJECTION_PATH:
            recurrent_extra.extend(["--svgd_kernel_projection_path", RECURRENT_SVGD_KERNEL_PROJECTION_PATH])
    for spec in specs:
        for score_target in configured_score_targets():
            public_target = public_score_target(score_target)
            data_jsonl = spec.data_jsonl
            output_suffix = public_target
            permutation_jsonl: Path | None = None
            if score_target in CYCLIC_LABEL_TARGETS:
                permutation_jsonl = cyclic_permutation_jsonl(spec)
                data_jsonl = permutation_jsonl
                output_suffix = "cyclic_label_raw"
            common = [
                sys.executable,
                "eval/eval_mcq.py",
                "--data_jsonl",
                path_for_cli(data_jsonl),
                "--prompt_style",
                prompt_style_for_score_target(score_target),
                "--score_target",
                eval_score_target(score_target),
                "--aggregates",
                AGGREGATES,
                "--dtype",
                DTYPE,
                "--adapter_dtype",
                ADAPTER_DTYPE,
                "--device",
                DEVICE,
                "--seed",
                "0",
            ]
            for arm, mode, extra in (
                ("base", "base", []),
                ("recurrent", RECURRENT_MODE, recurrent_extra),
            ):
                eval_output = RUN_DIR / f"{spec.name}_{arm}_{output_suffix}.jsonl"
                output = (
                    RUN_DIR / f"{spec.name}_{arm}_{public_target}.jsonl"
                    if permutation_jsonl is not None
                    else eval_output
                )
                loop_diagnostics = ["--include_loop_diagnostics"] if arm == "recurrent" and INCLUDE_LOOP_DIAGNOSTICS else []
                jobs.append(
                    EvalJob(
                        benchmark=spec.name,
                        arm=arm,
                        score_target=public_target,
                        output_jsonl=output,
                        cmd=common
                        + [
                            "--mode",
                            mode,
                            *extra,
                            *loop_diagnostics,
                            "--quiet_rows",
                            "--output_jsonl",
                            path_for_cli(eval_output),
                        ],
                        eval_output_jsonl=eval_output,
                        permutation_jsonl=permutation_jsonl,
                    )
                )
    return jobs


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aggregate.setdefault(str(row.get("aggregate") or "mean"), []).append(row)
    return {
        aggregate: {
            "correct": sum(1 for row in aggregate_rows if row.get("hit")),
            "total": len(aggregate_rows),
            "accuracy": sum(1 for row in aggregate_rows if row.get("hit")) / max(len(aggregate_rows), 1),
        }
        for aggregate, aggregate_rows in sorted(by_aggregate.items())
    }


def two_sided_sign_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows if row.get("id") is not None}


def compact_routing_diagnosis(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep aggregate routing evidence without storing prompt text examples."""

    return {
        "benchmark": summary["benchmark"],
        "paired_examples": summary["paired_examples"],
        "base_correct": summary["base_correct"],
        "candidate_correct": summary["candidate_correct"],
        "delta": summary["delta"],
        "changes": summary["changes"],
        "mean_margin_delta": summary["mean_margin_delta"],
        "mean_correct_score_delta": summary["mean_correct_score_delta"],
        "prediction_counts": summary["prediction_counts"],
        "features": summary["features"],
        "routing_buckets": summary["routing_buckets"],
    }


def routing_diagnostics(
    *,
    specs: list[BenchmarkSpec],
    raw_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    specs_by_name = {spec.name: spec for spec in specs}
    for benchmark, score_targets in raw_rows.items():
        spec = specs_by_name.get(benchmark)
        if spec is None:
            continue
        data_rows = mcq_rows_by_id(read_rows(spec.data_jsonl))
        for score_target, arms in score_targets.items():
            base_rows = arms.get("base") or []
            recurrent_rows = arms.get("recurrent") or []
            if not base_rows or not recurrent_rows:
                continue
            paired = paired_mcq_regression_rows(
                mcq_rows_by_id(base_rows),
                mcq_rows_by_id(recurrent_rows),
                data_rows,
            )
            diagnostics.setdefault(benchmark, {})[score_target] = compact_routing_diagnosis(
                summarize_mcq_regressions(paired, benchmark=benchmark)
            )
    return diagnostics


def loop_bucket_diagnostics(routing: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract loop-use telemetry from routing summaries.

    ``routing_diagnostics`` already has the information, but this top-level
    view makes the depth question visible in downstream assessments: are hard
    buckets actually routed deeper than direct buckets?
    """

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for benchmark, targets in routing.items():
        for score_target, summary in targets.items():
            buckets = {}
            for bucket, values in (summary.get("routing_buckets") or {}).items():
                buckets[bucket] = {
                    "n": values.get("n"),
                    "delta": values.get("delta"),
                    "wins": values.get("wins"),
                    "losses": values.get("losses"),
                    "mean_candidate_expected_loops": values.get("mean_candidate_expected_loops"),
                    "mean_candidate_answer_expected_loops": values.get("mean_candidate_answer_expected_loops"),
                    "mean_margin_delta": values.get("mean_margin_delta"),
                }
            out.setdefault(benchmark, {})[score_target] = {"routing_buckets": buckets}
    return out


def hard_content_signal(
    *,
    comparisons: dict[str, Any],
    paired_comparisons: dict[str, Any],
    routing: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize the thesis-relevant hard content-only surface."""

    signal: dict[str, Any] = {}
    for benchmark in ("arc_challenge", "open_hard_arc_challenge"):
        target = (comparisons.get(benchmark) or {}).get("content_question_only")
        if not target:
            continue
        aggregate = "mean"
        row = target.get(aggregate)
        if not row:
            continue
        signal[benchmark] = {
            "score_target": "content_question_only",
            "aggregate": aggregate,
            "base_correct": row["base"]["correct"],
            "recurrent_correct": row["recurrent"]["correct"],
            "total": row["recurrent"]["total"],
            "correct_delta_recurrent_vs_base": row["correct_delta_recurrent_vs_base"],
            "accuracy_delta_recurrent_vs_base": row["accuracy_delta_recurrent_vs_base"],
            "paired": (paired_comparisons.get(benchmark) or {}).get("content_question_only", {}).get(aggregate),
            "routing_buckets": (
                (routing.get(benchmark) or {}).get("content_question_only", {}).get("routing_buckets", {})
            ),
        }
    return signal


def paired_arm_summaries(
    base_rows: list[dict[str, Any]],
    recurrent_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    base_by_aggregate: dict[str, list[dict[str, Any]]] = {}
    recurrent_by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in base_rows:
        base_by_aggregate.setdefault(str(row.get("aggregate") or "mean"), []).append(row)
    for row in recurrent_rows:
        recurrent_by_aggregate.setdefault(str(row.get("aggregate") or "mean"), []).append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for aggregate in sorted(set(base_by_aggregate) | set(recurrent_by_aggregate)):
        base_by_key = rows_by_id(base_by_aggregate.get(aggregate, []))
        recurrent_by_key = rows_by_id(recurrent_by_aggregate.get(aggregate, []))
        common_ids = sorted(set(base_by_key) & set(recurrent_by_key))
        paired = [
            (bool(base_by_key[item].get("hit")), bool(recurrent_by_key[item].get("hit")))
            for item in common_ids
        ]
        base_correct = sum(int(base_hit) for base_hit, _recurrent_hit in paired)
        recurrent_correct = sum(int(recurrent_hit) for _base_hit, recurrent_hit in paired)
        wins = sum(1 for base_hit, recurrent_hit in paired if recurrent_hit and not base_hit)
        losses = sum(1 for base_hit, recurrent_hit in paired if base_hit and not recurrent_hit)
        ties = len(paired) - wins - losses
        summaries[aggregate] = {
            "paired_examples": len(paired),
            "base_correct": base_correct,
            "recurrent_correct": recurrent_correct,
            "correct_delta_recurrent_vs_base": recurrent_correct - base_correct,
            "base_accuracy": base_correct / max(len(paired), 1),
            "recurrent_accuracy": recurrent_correct / max(len(paired), 1),
            "accuracy_delta_recurrent_vs_base": (recurrent_correct - base_correct) / max(len(paired), 1),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_test_p_value": two_sided_sign_p_value(wins, losses),
        }
    return summaries


def compare_arm_summaries(base: dict[str, Any], recurrent: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for aggregate in sorted(set(base) | set(recurrent)):
        base_row = base.get(aggregate) or {"correct": 0, "total": 0, "accuracy": 0.0}
        recurrent_row = recurrent.get(aggregate) or {"correct": 0, "total": 0, "accuracy": 0.0}
        comparisons[aggregate] = {
            "correct_delta_recurrent_vs_base": int(recurrent_row["correct"]) - int(base_row["correct"]),
            "accuracy_delta_recurrent_vs_base": float(recurrent_row["accuracy"]) - float(base_row["accuracy"]),
            "base": base_row,
            "recurrent": recurrent_row,
        }
    return comparisons


def build_summary(
    *,
    source_summary: Path | None,
    checkpoint: Path,
    specs: list[BenchmarkSpec],
    jobs: list[EvalJob],
    failures: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    result_rows: dict[str, dict[str, Any]] = {}
    raw_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for job in jobs:
        rows = read_rows(job.output_jsonl)
        raw_rows.setdefault(job.benchmark, {}).setdefault(job.score_target, {})[job.arm] = rows
        result_rows.setdefault(job.benchmark, {}).setdefault(job.score_target, {})[job.arm] = summarize_rows(rows)
    comparisons: dict[str, Any] = {}
    paired_comparisons: dict[str, Any] = {}
    for benchmark, score_targets in result_rows.items():
        comparisons[benchmark] = {}
        for score_target, arms in score_targets.items():
            comparisons[benchmark][score_target] = compare_arm_summaries(
                arms.get("base") or {},
                arms.get("recurrent") or {},
            )
            raw_arms = raw_rows.get(benchmark, {}).get(score_target, {})
            paired_comparisons.setdefault(benchmark, {})[score_target] = paired_arm_summaries(
                raw_arms.get("base") or [],
                raw_arms.get("recurrent") or [],
            )
    routing = routing_diagnostics(specs=specs, raw_rows=raw_rows)
    loop_buckets = loop_bucket_diagnostics(routing)
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_benchmark_suite",
        "status": "completed_with_failures" if failures else "completed",
        "suite_profile": SUITE_PROFILE,
        "source_summary": path_for_cli(source_summary) if source_summary else None,
        "checkpoint": path_for_cli(checkpoint),
        "benchmarks": [spec.name for spec in specs],
        "score_targets": effective_score_targets(),
        "aggregates": parse_csv(AGGREGATES),
        "recurrent_mode": RECURRENT_MODE,
        "recurrent_num_trajectories": RECURRENT_NUM_TRAJECTORIES,
        "recurrent_use_learned_loop_control": RECURRENT_USE_LEARNED_LOOP_CONTROL,
        "elapsed_seconds": elapsed_seconds,
        "failures": failures,
        "results": result_rows,
        "comparisons": comparisons,
        "paired_comparisons": paired_comparisons,
        "routing_diagnostics": routing,
        "loop_bucket_diagnostics": loop_buckets,
        "hard_content_signal": hard_content_signal(
            comparisons=comparisons,
            paired_comparisons=paired_comparisons,
            routing=routing,
        ),
    }
    if AFTER_CONFIRM_DENSE_RUN_SUFFIX:
        payload["after_confirmation_dense_control"] = {
            "run_suffix": AFTER_CONFIRM_DENSE_RUN_SUFFIX,
            "extra_train_jsonl": AFTER_CONFIRM_DENSE_EXTRA_TRAIN_JSONL or None,
            "reason": "run dense same-curriculum control after this recurrent confirmation passes",
        }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_DIR / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_current_source_summary(summary_path)
    lines = [
        f"# Stage 5 Benchmark Suite - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Suite profile: `{payload.get('suite_profile')}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Benchmarks: `{payload['benchmarks']}`",
        f"- Recurrent mode: `{payload['recurrent_mode']}`",
        f"- Recurrent trajectories: `{payload['recurrent_num_trajectories']}`",
        f"- Learned loop control: `{payload.get('recurrent_use_learned_loop_control')}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']:.2f}`",
        "",
        "## Recurrent vs Base",
        "",
    ]
    for benchmark, score_targets in payload["comparisons"].items():
        lines.append(f"### {benchmark}")
        for score_target, aggregates in score_targets.items():
            lines.append(f"- score target `{score_target}`")
            for aggregate, row in aggregates.items():
                lines.append(
                    f"  - aggregate `{aggregate}`: correct delta "
                    f"`{row['correct_delta_recurrent_vs_base']}`, accuracy delta "
                    f"`{row['accuracy_delta_recurrent_vs_base']:.4f}` "
                    f"(base `{row['base']['correct']}/{row['base']['total']}`, "
                    f"recurrent `{row['recurrent']['correct']}/{row['recurrent']['total']}`)"
                )
            paired = payload.get("paired_comparisons", {}).get(benchmark, {}).get(score_target, {})
            if paired:
                lines.append("  - paired evidence")
                for aggregate, row in paired.items():
                    lines.append(
                        f"    - aggregate `{aggregate}`: recurrent `{row['recurrent_correct']}` / "
                        f"`{row['paired_examples']}`, base `{row['base_correct']}` / "
                        f"`{row['paired_examples']}`, delta "
                        f"`{row['correct_delta_recurrent_vs_base']}`, W/L/T "
                        f"`{row['wins']}/{row['losses']}/{row['ties']}`, p "
                        f"`{row['sign_test_p_value']}`"
                    )
            routing = payload.get("routing_diagnostics", {}).get(benchmark, {}).get(score_target, {})
            if routing:
                lines.append("  - routing buckets")
                for bucket, values in routing.get("routing_buckets", {}).items():
                    lines.append(
                        f"    - `{bucket}`: n `{values['n']}`, delta "
                        f"`{values['delta']}`, W/L `{values['wins']}/{values['losses']}`, "
                        f"mean margin delta `{values['mean_margin_delta']}`, "
                        f"mean loops `{values['mean_candidate_expected_loops']}`"
                    )
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in payload["failures"]:
            lines.append(f"- `{failure['stage']}` `{failure.get('benchmark')}`: {failure['error']}")
    if payload.get("hard_content_signal"):
        lines.extend(["", "## Hard Content Signal", ""])
        for benchmark, signal in payload["hard_content_signal"].items():
            lines.append(
                f"- `{benchmark}` `{signal['score_target']}`: delta "
                f"`{signal['correct_delta_recurrent_vs_base']}` "
                f"(base `{signal['base_correct']}/{signal['total']}`, "
                f"recurrent `{signal['recurrent_correct']}/{signal['total']}`)"
            )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "status", "-sb"], check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    pointer = current_source_summary_file()
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No benchmark suite outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 benchmark suite {RUN_ID} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode == 0:
        return
    print("Initial benchmark push failed; attempting one autostash rebase and retry.", flush=True)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def main() -> int:
    ensure_hf_token_from_colab()
    if RECURRENT_MODE == "phase1" and RECURRENT_NUM_TRAJECTORIES != 1:
        raise SystemExit("phase1 recurrent benchmark requires STAGE5_BENCHMARK_NUM_TRAJECTORIES=1")
    started = time.time()
    source_summary = resolve_source_summary()
    source_payload = read_json(source_summary) if source_summary else None
    resolved_source_summary = checkpoint_bearing_source_summary(source_summary, source_payload)
    if resolved_source_summary != source_summary:
        print(
            f"resolved_benchmark_source_summary={path_for_cli(source_summary)} -> "
            f"{path_for_cli(resolved_source_summary)}",
            flush=True,
        )
        source_summary = resolved_source_summary
        source_payload = read_json(source_summary) if source_summary else None
    checkpoint = resolve_checkpoint(source_summary, source_payload)
    specs = benchmark_specs(parse_csv(BENCHMARKS))
    failures: list[dict[str, Any]] = []

    for spec in specs:
        try:
            run(spec.prepare_cmd, log_name=f"prepare_{spec.name}.log")
        except Exception as exc:
            failures.append({"stage": "prepare", "benchmark": spec.name, "error": str(exc)})
            if not CONTINUE_ON_FAILURE:
                raise

    jobs = [job for job in eval_jobs(specs, checkpoint=checkpoint) if job.output_jsonl.parent.exists()]
    for job in jobs:
        if not job.cmd[job.cmd.index("--data_jsonl") + 1]:
            continue
        data_path = resolve_path(job.cmd[job.cmd.index("--data_jsonl") + 1])
        if not data_path.exists():
            continue
        eval_output = job.eval_output_jsonl or job.output_jsonl
        if eval_output.exists():
            eval_output.unlink()
        if job.output_jsonl.exists():
            job.output_jsonl.unlink()
        try:
            run(job.cmd, log_name=f"{eval_output.stem}.log")
            aggregate_cyclic_label_output(job)
        except Exception as exc:
            failures.append(
                {
                    "stage": "eval",
                    "benchmark": job.benchmark,
                    "arm": job.arm,
                    "score_target": job.score_target,
                    "error": str(exc),
                }
            )
            if not CONTINUE_ON_FAILURE:
                raise

    payload = build_summary(
        source_summary=source_summary,
        checkpoint=checkpoint,
        specs=specs,
        jobs=jobs,
        failures=failures,
        elapsed_seconds=time.time() - started,
    )
    write_report(payload)
    commit_results()
    if failures and not CONTINUE_ON_FAILURE:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
