"""Run a bounded Stage 5 depth/width routing diagnostic.

This is the next credit-aware GPU action after the failed ARC-mix recovery
proxy. It does not train. It restores the best recovered deterministic Phase 1
checkpoint, runs a small ARC-Easy plus ARC-Challenge MCQ benchmark with loop
diagnostics enabled, and writes a routing assessment that tells us whether the
next spend should be direct-mode halting repair, deep-narrow Phase 1 recovery,
or a larger confirmation.

Defaults are intentionally modest so this can run on L4/T4 when available. Use
an A100 only if it is already the practical runtime choice, not because this job
needs A100 headroom.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    DEFAULT_CHECKPOINT_REL,
    DEFAULT_RECOVERED_RUN_ID,
    DEFAULT_SOURCE_SUMMARY_REL,
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_ROUTING_DIAGNOSTIC_RUN_ID") or time.strftime(
    "stage5_routing_diagnostic_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

PUSH_RESULTS = os.environ.get("STAGE5_ROUTING_DIAGNOSTIC_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), "".join(chunks), None)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def default_benchmark_run_id(easy_limit: str, challenge_limit: str) -> str:
    easy = easy_limit.strip().lower().replace("none", "full").replace("all", "full")
    challenge = challenge_limit.strip().lower().replace("none", "full").replace("all", "full")
    return f"{RUN_ID}_arc_easy{easy}_challenge{challenge}"


def build_benchmark_env(
    *,
    checkpoint: Path,
    source_summary: Path | None,
    easy_limit: str,
    challenge_limit: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("STAGE5_BENCHMARK_SUITE_RUN_ID", default_benchmark_run_id(easy_limit, challenge_limit))
    env["STAGE5_BENCHMARK_CHECKPOINT"] = path_for_cli(checkpoint)
    if source_summary and source_summary.exists():
        env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env["STAGE5_BENCHMARKS"] = "arc_easy,arc_challenge"
    env["STAGE5_BENCHMARK_ARC_EASY_LIMIT"] = easy_limit
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = challenge_limit
    env["STAGE5_BENCHMARK_SCORE_TARGETS"] = os.environ.get("STAGE5_BENCHMARK_SCORE_TARGETS", "label")
    env["STAGE5_BENCHMARK_AGGREGATES"] = os.environ.get("STAGE5_BENCHMARK_AGGREGATES", "mean")
    env["STAGE5_BENCHMARK_RECURRENT_MODE"] = "phase1"
    env["STAGE5_BENCHMARK_NUM_TRAJECTORIES"] = "1"
    env["STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS"] = "1"
    env["STAGE5_BENCHMARK_CONTINUE_ON_FAILURE"] = "0"
    env["STAGE5_BENCHMARK_PUSH"] = "0"
    env.setdefault("DTYPE", "bfloat16")
    env.setdefault("ADAPTER_DTYPE", "float32")
    env.setdefault("DEVICE", "cuda")
    return env


def bucket_metric(bucket: dict[str, Any] | None, key: str) -> float | None:
    if not bucket:
        return None
    value = bucket.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def routing_rollup(summary: dict[str, Any]) -> dict[str, Any]:
    rollup: dict[str, Any] = {}
    for benchmark, score_targets in (summary.get("routing_diagnostics") or {}).items():
        target_payload = score_targets.get("label") or next(iter(score_targets.values()), {})
        buckets = target_payload.get("routing_buckets") or {}
        direct = buckets.get("base_confident_direct_proxy") or {}
        deep = buckets.get("deep_numeric_proxy") or {}
        conceptual = buckets.get("conceptual_reasoning_proxy") or {}
        rollup[benchmark] = {
            "paired_examples": target_payload.get("paired_examples", 0),
            "delta": target_payload.get("delta", 0),
            "direct": direct,
            "deep_numeric": deep,
            "conceptual": conceptual,
            "direct_delta": bucket_metric(direct, "delta"),
            "direct_mean_loops": bucket_metric(direct, "mean_candidate_expected_loops"),
            "direct_mean_margin_delta": bucket_metric(direct, "mean_margin_delta"),
            "deep_delta": bucket_metric(deep, "delta"),
            "deep_mean_loops": bucket_metric(deep, "mean_candidate_expected_loops"),
            "conceptual_delta": bucket_metric(conceptual, "delta"),
        }
    return rollup


def any_negative_direct_drift(rollup: dict[str, Any]) -> bool:
    for item in rollup.values():
        direct_delta = item.get("direct_delta")
        margin_delta = item.get("direct_mean_margin_delta")
        direct_loops = item.get("direct_mean_loops")
        if direct_delta is not None and direct_delta < 0:
            return True
        if margin_delta is not None and margin_delta < -0.05:
            return True
        if direct_loops is not None and direct_loops > 2.0:
            return True
    return False


def any_deep_nonpositive(rollup: dict[str, Any]) -> bool:
    seen_deep = False
    for item in rollup.values():
        deep_delta = item.get("deep_delta")
        if deep_delta is None:
            continue
        seen_deep = True
        if deep_delta < 0:
            return True
    return not seen_deep


def assess(summary: dict[str, Any]) -> dict[str, Any]:
    rollup = routing_rollup(summary)
    if any_negative_direct_drift(rollup):
        status = "needs_direct_halting_repair"
        next_action = "Train Phase 1 direct-mode recovery with base-logit distillation and shallow halt supervision."
    elif any_deep_nonpositive(rollup):
        status = "needs_deep_narrow_recovery"
        next_action = "Train Phase 1 deep-narrow recovery with repulsion off and non-collapsed halt-depth targets."
    else:
        status = "routing_diagnostic_pass"
        next_action = "Run a larger confirmation or proceed to the bounded direct/deep recovery ladder."
    return {
        "run_id": RUN_ID,
        "kind": "stage5_routing_diagnostic_assessment",
        "benchmark_summary": path_for_cli(RUN_DIR / "benchmark_run" / "summary.json"),
        "status": status,
        "next_action": next_action,
        "rollup": rollup,
    }


def write_report(payload: dict[str, Any]) -> None:
    write_json(RUN_DIR / "routing_assessment.json", payload)
    write_json(RUN_DIR / "summary.json", payload)
    lines = [
        f"# Stage 5 Routing Diagnostic - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Next action: {payload['next_action']}",
        "",
        "## Routing Rollup",
        "",
        "| Benchmark | paired | delta | direct delta | direct loops | direct margin delta | deep delta | deep loops | conceptual delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, row in payload["rollup"].items():
        lines.append(
            "| {benchmark} | {paired} | {delta} | {direct_delta} | {direct_loops} | {direct_margin} | {deep_delta} | {deep_loops} | {conceptual_delta} |".format(
                benchmark=benchmark,
                paired=row.get("paired_examples"),
                delta=row.get("delta"),
                direct_delta=row.get("direct_delta"),
                direct_loops=row.get("direct_mean_loops"),
                direct_margin=row.get("direct_mean_margin_delta"),
                deep_delta=row.get("deep_delta"),
                deep_loops=row.get("deep_mean_loops"),
                conceptual_delta=row.get("conceptual_delta"),
            )
        )
    (RUN_DIR / "routing_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "routing_assessment.md").read_text(encoding="utf-8"))


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "status", "-sb"], check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No routing diagnostic outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 routing diagnostic {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    recovered_run_id = os.environ.get("STAGE5_RECOVERED_PHASE1_RUN_ID", DEFAULT_RECOVERED_RUN_ID)
    checkpoint = resolve_root_path(os.environ.get("STAGE5_RECOVERED_PHASE1_CHECKPOINT", DEFAULT_CHECKPOINT_REL))
    source_summary = resolve_root_path(os.environ.get("STAGE5_RECOVERED_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY_REL))
    easy_limit = os.environ.get("STAGE5_ROUTING_ARC_EASY_LIMIT", "64")
    challenge_limit = os.environ.get("STAGE5_ROUTING_ARC_CHALLENGE_LIMIT", "64")

    restore_checkpoint_if_needed(checkpoint, run_id=recovered_run_id)

    env = build_benchmark_env(
        checkpoint=checkpoint,
        source_summary=source_summary if source_summary.exists() else None,
        easy_limit=easy_limit,
        challenge_limit=challenge_limit,
    )
    print(f"routing_run_id={RUN_ID}")
    print(f"checkpoint={path_for_cli(checkpoint)}")
    print(f"arc_easy_limit={easy_limit}")
    print(f"arc_challenge_limit={challenge_limit}")
    print(f"benchmark_run_id={env['STAGE5_BENCHMARK_SUITE_RUN_ID']}")

    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env)
    benchmark_run_dir = ROOT / "outputs" / "stage5" / env["STAGE5_BENCHMARK_SUITE_RUN_ID"]
    benchmark_summary = benchmark_run_dir / "summary.json"
    if not benchmark_summary.exists():
        raise FileNotFoundError(benchmark_summary)
    benchmark_payload = read_json(benchmark_summary)
    shutil.copytree(benchmark_run_dir, RUN_DIR / "benchmark_run", dirs_exist_ok=True)
    payload = assess(benchmark_payload)
    write_report(payload)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
