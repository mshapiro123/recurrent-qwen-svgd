"""Plan the no-GPU curriculum scale-up that runs beside re-entry repair.

This script does not generate rows or call a provider. It reads the latest
gate-ready capability-ladder trace collection, compares it against the current
claim-sized direct/deep objective, and prints concrete next commands. The point
is to keep the paid-GPU queue focused on Phase 0/1 while CPU/API work fills the
Stage 4 curriculum deficit in parallel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.reentry_recovery_config import (  # noqa: E402
    DEFAULT_CLAIM_MIN_TARGET_LOOP_ROWS,
    DEFAULT_CLAIM_MIN_MODE_ROWS,
    DEFAULT_CLAIM_MIN_POSITIVE_ROWS,
    assess_trace_curriculum_for_reentry_recovery,
    parse_row_requirements,
)
from colab.review_stage5_recovery_curriculum import (  # noqa: E402
    path_for_cli,
    read_json,
    resolve_trace_collection_summary,
)


DEFAULT_WORK_DIR = "data/curriculum/claim_direct_deep_001"
DEFAULT_MODEL_MAP = "data/curriculum/claim_direct_deep_001/model_map.json"


def int_arg(value: str, *, minimum: int = 0) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"Expected integer >= {minimum}, got {value!r}.")
    return parsed


def compact_requirements(requirements: dict[str, int]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(requirements.items()))


def mode_deficits(claim: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    report = claim.get("mode_requirements") if isinstance(claim.get("mode_requirements"), dict) else {}
    for key, payload in report.items():
        if isinstance(payload, dict):
            deficit = int(payload.get("deficit") or 0)
            if deficit > 0:
                out[str(key)] = deficit
    return out


def observed_modes(assessment: dict[str, Any]) -> dict[str, int]:
    counts = assessment.get("counts") if isinstance(assessment.get("counts"), dict) else {}
    modes = counts.get("mode_counts") if isinstance(counts.get("mode_counts"), dict) else {}
    return {str(key): int(value) for key, value in modes.items()}


def estimate_count_per_combo(
    *,
    positive_deficit: int,
    seed_models: int,
    domains: int,
    difficulties: int,
    target_steps: int,
    overgenerate_factor: float,
) -> int:
    combos = max(1, seed_models * domains * difficulties * target_steps)
    target_seed_jobs = max(1, math.ceil(positive_deficit * overgenerate_factor))
    return max(1, math.ceil(target_seed_jobs / combos))


def command_text(parts: list[str]) -> str:
    return " ".join(parts)


def build_plan(
    *,
    trace_summary: Path,
    assessment: dict[str, Any],
    work_dir: str,
    provider_backend: str,
    api_key_env: str,
    model_map: str,
    overgenerate_factor: float,
    claim_min_positive_rows: int,
    claim_min_mode_rows: str,
    claim_min_target_loop_rows: str = "",
) -> dict[str, Any]:
    claim = assessment["claim_readiness"]
    positive_deficit = int(claim.get("positive_row_deficit") or 0)
    requirements = parse_row_requirements(claim_min_mode_rows)
    deficits = mode_deficits(claim)
    observed = observed_modes(assessment)

    # The artifact pipeline defaults are kept in sync with
    # CURRICULUM_ARTIFACT_PIPELINE_CELL.py but made explicit in the plan so the
    # user can estimate provider spend before running it.
    seed_models = ["opus-strong", "glm-strong"]
    domains = ["math", "science"]
    difficulties = ["medium", "hard"]
    target_steps = ["1", "2", "5", "9"]
    count_per_combo = estimate_count_per_combo(
        positive_deficit=max(positive_deficit, sum(deficits.values()), 1),
        seed_models=len(seed_models),
        domains=len(domains),
        difficulties=len(difficulties),
        target_steps=len(target_steps),
        overgenerate_factor=overgenerate_factor,
    )

    pipeline_cmd = command_text(
        [
            "python",
            "training/run_curriculum_pipeline_from_artifacts.py",
            "--work_dir",
            work_dir,
            "--seed_models",
            ",".join(seed_models),
            "--solver_models",
            ",".join(seed_models),
            "--judge_models",
            ",".join(seed_models),
            "--domains",
            ",".join(domains),
            "--difficulties",
            ",".join(difficulties),
            "--target_steps",
            ",".join(target_steps),
            "--count_per_combo",
            str(count_per_combo),
            "--reference_model",
            "weak-reference",
            "--reference_samples",
            "3",
            "--min_reference_samples",
            "1",
            "--min_natural_agree",
            "2",
            "--min_distinct_agree",
            "2",
            "--require_programmatic_answer_check",
        ]
    )
    response_cmd = command_text(
        [
            "python",
            "training/run_curriculum_job_responses.py",
            "--jobs_jsonl",
            "<pending jobs_jsonl from summary.json>",
            "--output_jsonl",
            "<matching responses_jsonl from summary.json>",
            "--backend",
            provider_backend,
            "--api_key_env",
            api_key_env,
            "--model_map_json",
            model_map,
            "--resume",
            "--max_retries",
            "3",
            "--retry_sleep_sec",
            "2",
            "--retry_backoff",
            "2",
            "--json_mode",
        ]
    )
    gate_parts = [
        "python",
        "training/check_curriculum_sft_gate.py",
        "--work_dir",
        work_dir,
        "--min_positive_rows",
        str(claim_min_positive_rows),
        "--min_mode_rows",
        compact_requirements(requirements),
    ]
    if claim_min_target_loop_rows:
        gate_parts += ["--min_target_loop_rows", claim_min_target_loop_rows]
    gate_parts.append("--fail_on_no_go")
    gate_cmd = command_text(gate_parts)

    actions = [
        {
            "name": "keep_gpu_on_phase0",
            "runtime": "L4/T4 GPU",
            "command": "STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke",
            "why": "The master sequence says loop closure is the current blocker; do not spend GPU on Stage 4 until Stage 3 passes.",
        },
        {
            "name": "scale_cpu_api_curriculum",
            "runtime": "CPU or cheap non-GPU runtime",
            "command": "STAGE5_CURRENT_A100_TARGET=claim_curriculum_scaleup_cpu",
            "why": "Use the maintained bootstrap target for the claim-sized direct/deep artifact pipeline.",
        },
        {
            "name": "scale_cpu_api_curriculum_raw",
            "runtime": "CPU or cheap non-GPU runtime",
            "command": pipeline_cmd,
            "why": "Underlying command run by the target when you need to debug locally.",
        },
        {
            "name": "fill_pending_provider_responses",
            "runtime": "CPU/network runtime",
            "command": response_cmd,
            "why": "Run only the pending job/response pair reported by the pipeline summary, then rerun the pipeline command.",
        },
        {
            "name": "claim_sized_gate",
            "runtime": "CPU",
            "command": gate_cmd,
            "why": "Do not use this data for a performance claim until direct and deep-narrow quotas are green.",
        },
    ]
    return {
        "kind": "stage5_curriculum_scaleup_plan",
        "trace_summary": path_for_cli(trace_summary),
        "current_counts": assessment.get("counts", {}),
        "bounded_smoke": {
            "go": assessment.get("go"),
            "status": assessment.get("status"),
            "warnings": assessment.get("warnings", []),
            "next_step": assessment.get("next_step"),
        },
        "claim_readiness": claim,
        "observed_modes": observed,
        "mode_deficits": deficits,
        "positive_deficit": positive_deficit,
        "work_dir": work_dir,
        "generation_assumptions": {
            "seed_models": seed_models,
            "domains": domains,
            "difficulties": difficulties,
            "target_steps": target_steps,
            "overgenerate_factor": overgenerate_factor,
            "estimated_count_per_combo": count_per_combo,
        },
        "actions": actions,
        "phase_order_warning": (
            "This data plan runs in parallel with Phase 0. It does not unlock Stage 4; "
            "Stage 3 must still publish a repair assessment recommending bounded recovery training."
        ),
    }


def print_markdown(plan: dict[str, Any]) -> None:
    claim = plan["claim_readiness"]
    generation = plan["generation_assumptions"]
    print("# Stage 5 Curriculum Scale-Up Plan")
    print()
    print(f"- Trace summary: `{plan['trace_summary']}`")
    print(f"- Bounded Stage 4 smoke ready: `{plan['bounded_smoke']['go']}`")
    print(f"- Claim-sized ready: `{claim.get('go')}`")
    print(f"- Positive row deficit: `{plan['positive_deficit']}`")
    print(f"- Mode deficits: `{plan['mode_deficits']}`")
    print(f"- Current counts: `{plan['current_counts']}`")
    print(f"- Work dir: `{plan['work_dir']}`")
    print(f"- Estimated count_per_combo: `{generation['estimated_count_per_combo']}`")
    print()
    print("## Actions")
    print()
    for index, action in enumerate(plan["actions"], start=1):
        print(f"{index}. **{action['name']}** ({action['runtime']})")
        print()
        print(f"   Why: {action['why']}")
        print()
        print("   ```bash")
        print(f"   {action['command']}")
        print("   ```")
        print()
    print("## Warning")
    print()
    print(plan["phase_order_warning"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-summary", default="")
    parser.add_argument("--claim-min-positive-rows", type=int, default=DEFAULT_CLAIM_MIN_POSITIVE_ROWS)
    parser.add_argument("--claim-min-mode-rows", default=compact_requirements(DEFAULT_CLAIM_MIN_MODE_ROWS))
    parser.add_argument("--claim-min-target-loop-rows", default=compact_requirements(DEFAULT_CLAIM_MIN_TARGET_LOOP_ROWS))
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--provider-backend", default="openai_compatible")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model-map-json", default=DEFAULT_MODEL_MAP)
    parser.add_argument("--overgenerate-factor", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    explicit = (
        args.trace_summary
        or os.environ.get("STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY")
        or os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY")
        or ""
    )
    trace_summary = resolve_trace_collection_summary(explicit=explicit)
    assessment = assess_trace_curriculum_for_reentry_recovery(
        read_json(trace_summary),
        claim_min_positive_rows=args.claim_min_positive_rows,
        claim_min_mode_rows=args.claim_min_mode_rows,
        claim_min_target_loop_rows=args.claim_min_target_loop_rows,
    )
    plan = build_plan(
        trace_summary=trace_summary,
        assessment=assessment,
        work_dir=args.work_dir,
        provider_backend=args.provider_backend,
        api_key_env=args.api_key_env,
        model_map=args.model_map_json,
        overgenerate_factor=args.overgenerate_factor,
        claim_min_positive_rows=args.claim_min_positive_rows,
        claim_min_mode_rows=args.claim_min_mode_rows,
        claim_min_target_loop_rows=args.claim_min_target_loop_rows,
    )
    if args.json:
        print(json.dumps(plan, indent=2), flush=True)
    else:
        print_markdown(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
