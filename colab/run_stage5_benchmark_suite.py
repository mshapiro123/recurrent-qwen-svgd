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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_publish_hf_adapter import checkpoint_value_from_payload
from eval.analyze_mcq_regressions import (
    paired_rows as paired_mcq_regression_rows,
    rows_by_id as mcq_rows_by_id,
    summarize as summarize_mcq_regressions,
)


RUN_ID = os.environ.get("STAGE5_BENCHMARK_SUITE_RUN_ID") or time.strftime(
    "stage5_benchmark_suite_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DATA_DIR = ROOT / "data" / "stage5_benchmark_suite" / RUN_ID

SOURCE_SUMMARY = os.environ.get("STAGE5_BENCHMARK_SOURCE_SUMMARY", "")
EXPLICIT_CHECKPOINT = os.environ.get("STAGE5_BENCHMARK_CHECKPOINT", "")
BENCHMARKS = os.environ.get("STAGE5_BENCHMARKS", "arc_challenge,gpqa_lite")
ARC_CHALLENGE_LIMIT_RAW = os.environ.get("STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT", "128")
ARC_EASY_LIMIT_RAW = os.environ.get("STAGE5_BENCHMARK_ARC_EASY_LIMIT", "128")
GPQA_LIMIT = int(os.environ.get("STAGE5_BENCHMARK_GPQA_LIMIT", "16"))
GPQA_CONFIG = os.environ.get("STAGE5_BENCHMARK_GPQA_CONFIG", "gpqa_diamond")
SCORE_TARGETS = os.environ.get("STAGE5_BENCHMARK_SCORE_TARGETS", "label")
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


def resolve_checkpoint(source_summary: Path | None, payload: dict[str, Any] | None) -> Path:
    for candidate in checkpoint_candidates_from_payload(source_summary, payload):
        if candidate.exists():
            return candidate
    searched = [path_for_cli(path) for path in checkpoint_candidates_from_payload(source_summary, payload)]
    raise FileNotFoundError(f"No recurrent checkpoint found. Searched: {searched}")


def benchmark_specs(names: list[str]) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    for name in names:
        if name in {"arc_challenge", "arc_easy"}:
            config = "ARC-Challenge" if name == "arc_challenge" else "ARC-Easy"
            limit = ARC_CHALLENGE_LIMIT if name == "arc_challenge" else ARC_EASY_LIMIT
            limit_label = "full" if limit is None else str(limit)
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
            if limit is not None:
                prepare_cmd.extend(["--limit", str(limit)])
            output = PRIVATE_DATA_DIR / f"arc_challenge_validation_{limit_label}.jsonl"
            if name == "arc_easy":
                output = PRIVATE_DATA_DIR / f"arc_easy_validation_{limit_label}.jsonl"
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
            raise ValueError(f"Unknown benchmark {name!r}; expected arc_challenge, arc_easy, or gpqa_lite")
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
        for score_target in parse_csv(SCORE_TARGETS):
            common = [
                sys.executable,
                "eval/eval_mcq.py",
                "--data_jsonl",
                path_for_cli(spec.data_jsonl),
                "--prompt_style",
                "with_options",
                "--score_target",
                score_target,
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
                output = RUN_DIR / f"{spec.name}_{arm}_{score_target}.jsonl"
                loop_diagnostics = ["--include_loop_diagnostics"] if arm == "recurrent" and INCLUDE_LOOP_DIAGNOSTICS else []
                jobs.append(
                    EvalJob(
                        benchmark=spec.name,
                        arm=arm,
                        score_target=score_target,
                        output_jsonl=output,
                        cmd=common + ["--mode", mode, *extra, *loop_diagnostics, "--output_jsonl", path_for_cli(output)],
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
    return {
        "run_id": RUN_ID,
        "kind": "stage5_benchmark_suite",
        "status": "completed_with_failures" if failures else "completed",
        "source_summary": path_for_cli(source_summary) if source_summary else None,
        "checkpoint": path_for_cli(checkpoint),
        "benchmarks": [spec.name for spec in specs],
        "score_targets": parse_csv(SCORE_TARGETS),
        "aggregates": parse_csv(AGGREGATES),
        "recurrent_mode": RECURRENT_MODE,
        "recurrent_num_trajectories": RECURRENT_NUM_TRAJECTORIES,
        "elapsed_seconds": elapsed_seconds,
        "failures": failures,
        "results": result_rows,
        "comparisons": comparisons,
        "paired_comparisons": paired_comparisons,
        "routing_diagnostics": routing,
    }


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Benchmark Suite - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Benchmarks: `{payload['benchmarks']}`",
        f"- Recurrent mode: `{payload['recurrent_mode']}`",
        f"- Recurrent trajectories: `{payload['recurrent_num_trajectories']}`",
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
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "status", "-sb"], check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No benchmark suite outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 benchmark suite {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if RECURRENT_MODE == "phase1" and RECURRENT_NUM_TRAJECTORIES != 1:
        raise SystemExit("phase1 recurrent benchmark requires STAGE5_BENCHMARK_NUM_TRAJECTORIES=1")
    started = time.time()
    source_summary = resolve_source_summary()
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
        if job.output_jsonl.exists():
            job.output_jsonl.unlink()
        try:
            run(job.cmd, log_name=f"{job.output_jsonl.stem}.log")
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
