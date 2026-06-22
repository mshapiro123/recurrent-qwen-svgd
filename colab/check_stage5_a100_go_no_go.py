"""No-GPU go/no-go check before spending A100 credits on Stage 5.

This script reads a Stage 5 summary, asks the normal planner for the next
action, and classifies that action from the perspective of paid GPU use. It is
intentionally conservative: inspection actions and calibration warnings are
``no_go``; the current failed full ARC assessment maps to one bounded ARC-mix
proxy; a clean ARC-mix proxy pass maps to one full balanced confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.plan_stage5_next_run import (
    path_for_cli,
    plan_next_actions,
    read_json,
    resolve_path,
    resolve_source_summary,
    source_kind,
)
from colab.run_stage5_recovered_phase1_arc_gate import (
    DEFAULT_CHECKPOINT_REL,
    DEFAULT_RECOVERED_RUN_ID,
    candidate_drive_checkpoints,
    path_for_cli as checkpoint_path_for_cli,
)


RUN_ID = os.environ.get("STAGE5_A100_GO_NO_GO_RUN_ID") or time.strftime(
    "stage5_a100_go_no_go_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
STAGE4_OPUS_APPROVED_SOURCE_KEYS = {"opus47_sft", "opus47_raw"}
DEFAULT_CURRICULUM_MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"
DEFAULT_TRACE_CURRICULUM_MIN_SFT_ROWS = int(os.environ.get("STAGE5_TRACE_CURRICULUM_MIN_SFT_ROWS", "16"))
CURRICULUM_SFT_MIN_MEAN_EXPECTED_LOOPS = float(
    os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_CURRICULUM_SFT_MIN_MEAN_EXPECTED_LOOPS", "1.05")
)
DEFAULT_STAGE5_PHASE1_CHECKPOINT = (
    Path("outputs")
    / "stage4"
    / "stage4_opus_a100_20260620"
    / "phase1"
    / "phase1_step_500.pt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        default=None,
        help="Stage 5 summary JSON. Defaults to the latest planner-discoverable summary.",
    )
    return parser.parse_args(argv)


def command_script(command: str) -> str:
    parts = command.replace("\\", "/").split()
    for index, token in enumerate(parts):
        if token == "python" and index + 1 < len(parts):
            return parts[index + 1]
    if parts and parts[0] == "cat":
        return "cat"
    return ""


def command_env_assignments(command: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for token in shlex.split(command):
        if token == "python" or token == "cat":
            break
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and key.upper() == key:
            env[key] = value
    return env


@contextmanager
def scoped_environ(updates: dict[str, str]):
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_set(value: str | None) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _parse_benchmark_limit(value: str | None) -> tuple[int | None, str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None, "missing"
    if raw in {"none", "full", "all", "unbounded", "-1"}:
        return None, "unbounded"
    try:
        limit = int(raw)
    except ValueError:
        return None, "invalid"
    if limit <= 0:
        return None, "invalid"
    return limit, "bounded"


def benchmark_suite_budget_preflight(command: str) -> dict[str, Any]:
    """Conservatively require explicit bounded benchmark limits before A100 spend."""

    env = command_env_assignments(command)
    allow_full = _truthy_env("STAGE5_A100_ALLOW_FULL_BENCHMARKS")
    benchmarks = _csv_set(env.get("STAGE5_BENCHMARKS"))
    if not benchmarks:
        return {
            "go": False,
            "reason": (
                "Benchmark suite commands must declare STAGE5_BENCHMARKS and explicit "
                "per-benchmark limits before paid GPU spend."
            ),
            "limits": {},
        }

    limit_specs = {
        "arc_challenge": ("STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT", 256),
        "arc_easy": ("STAGE5_BENCHMARK_ARC_EASY_LIMIT", 256),
        "gpqa_lite": ("STAGE5_BENCHMARK_GPQA_LIMIT", 32),
    }
    limits: dict[str, Any] = {}
    failures: list[str] = []
    for benchmark in sorted(benchmarks):
        spec = limit_specs.get(benchmark)
        if spec is None:
            failures.append(f"{benchmark}: unsupported benchmark for A100 budget preflight")
            continue
        env_name, max_limit = spec
        value, status = _parse_benchmark_limit(env.get(env_name))
        limits[benchmark] = {
            "env": env_name,
            "raw": env.get(env_name),
            "status": status,
            "value": value,
            "max_without_override": max_limit,
        }
        if status == "missing":
            failures.append(f"{benchmark}: missing {env_name}")
        elif status == "invalid":
            failures.append(f"{benchmark}: invalid {env_name}={env.get(env_name)!r}")
        elif status == "unbounded" and not allow_full:
            failures.append(f"{benchmark}: unbounded {env_name} requires STAGE5_A100_ALLOW_FULL_BENCHMARKS=1")
        elif value is not None and value > max_limit and not allow_full:
            failures.append(
                f"{benchmark}: {env_name}={value} exceeds conservative cap {max_limit}; "
                "set STAGE5_A100_ALLOW_FULL_BENCHMARKS=1 for deliberate expansion"
            )

    return {
        "go": not failures,
        "reason": "Benchmark suite limits are explicit and bounded." if not failures else "; ".join(failures),
        "benchmarks": sorted(benchmarks),
        "limits": limits,
        "allow_full_override": allow_full,
    }


def capability_ladder_probe_budget_preflight(command: str) -> dict[str, Any]:
    """Require a bounded ARC slice before spending GPU on model-scale scoring."""

    env = command_env_assignments(command)
    allow_large = _truthy_env("STAGE5_A100_ALLOW_LARGE_CAPABILITY_LADDER_PROBE")
    raw_limit = env.get("STAGE5_CAPABILITY_LADDER_ARC_LIMIT")
    value, status = _parse_benchmark_limit(raw_limit)
    max_limit = 96
    failures: list[str] = []
    if status == "missing":
        failures.append("missing STAGE5_CAPABILITY_LADDER_ARC_LIMIT")
    elif status == "invalid":
        failures.append(f"invalid STAGE5_CAPABILITY_LADDER_ARC_LIMIT={raw_limit!r}")
    elif status == "unbounded" and not allow_large:
        failures.append(
            "unbounded STAGE5_CAPABILITY_LADDER_ARC_LIMIT requires "
            "STAGE5_A100_ALLOW_LARGE_CAPABILITY_LADDER_PROBE=1"
        )
    elif value is not None and value > max_limit and not allow_large:
        failures.append(
            f"STAGE5_CAPABILITY_LADDER_ARC_LIMIT={value} exceeds conservative cap {max_limit}; "
            "set STAGE5_A100_ALLOW_LARGE_CAPABILITY_LADDER_PROBE=1 for deliberate expansion"
        )

    return {
        "go": not failures,
        "reason": (
            "Capability-ladder probe limit is explicit and bounded."
            if not failures
            else "; ".join(failures)
        ),
        "limit": {
            "env": "STAGE5_CAPABILITY_LADDER_ARC_LIMIT",
            "raw": raw_limit,
            "status": status,
            "value": value,
            "max_without_override": max_limit,
        },
        "allow_large_override": allow_large,
    }


def source_payload_kind(source_payload: dict[str, Any]) -> str:
    if source_payload.get("kind"):
        return source_kind(source_payload)
    explicit = source_payload.get("source_kind")
    if explicit:
        return str(explicit)
    return source_kind(source_payload)


def default_phase1_checkpoint(command: str) -> str:
    env = command_env_assignments(command)
    return str(env.get("STAGE5_PHASE1_CKPT") or DEFAULT_STAGE5_PHASE1_CHECKPOINT).replace("\\", "/")


def recovered_checkpoint_from_command(command: str) -> str | None:
    env = command_env_assignments(command)
    for key in (
        "STAGE5_ARC_AGI_RECOVERED_CKPT",
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_SOURCE_CHECKPOINT",
        "STAGE5_PHASE1_CKPT",
    ):
        value = env.get(key)
        if value:
            return str(value).replace("\\", "/")
    return None


def go_paid_gpu_action(
    *,
    status: str,
    spend_class: str,
    reason: str,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    decision = {
        "go": True,
        "status": status,
        "spend_class": spend_class,
        "reason": reason,
    }
    if checkpoint:
        decision["checkpoint"] = checkpoint
    return decision


def normalize_min_mode_rows(value: str) -> str:
    items: list[tuple[str, int]] = []
    for raw in str(value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            mode, count = item.split("=", 1)
        elif ":" in item:
            mode, count = item.split(":", 1)
        else:
            return ""
        try:
            parsed = int(count.strip())
        except ValueError:
            return ""
        if parsed > 0:
            items.append((mode.strip(), parsed))
    return ",".join(f"{mode}={count}" for mode, count in sorted(items))


def source_curriculum_min_mode_rows(source_payload: dict[str, Any]) -> str:
    if source_payload.get("kind") == "stage5_capability_ladder_trace_collection":
        curriculum = source_payload.get("curriculum") if isinstance(source_payload.get("curriculum"), dict) else {}
        counts = curriculum.get("counts") if isinstance(curriculum.get("counts"), dict) else {}
        mode_counts = counts.get("mode_counts") if isinstance(counts.get("mode_counts"), dict) else {}
        parsed = normalize_min_mode_rows(
            ",".join(f"{mode}={int(count or 0)}" for mode, count in mode_counts.items())
        )
        if parsed:
            return parsed
    mode_requirements = ((source_payload.get("checks") or {}).get("positive_sft") or {}).get("mode_requirements")
    if isinstance(mode_requirements, dict):
        items = []
        for mode, payload in mode_requirements.items():
            if not isinstance(payload, dict):
                continue
            count = int(payload.get("required") or 0)
            if count > 0:
                items.append(f"{mode}={count}")
        parsed = normalize_min_mode_rows(",".join(items))
        if parsed:
            return parsed
    return normalize_min_mode_rows(os.environ.get("STAGE5_CURRICULUM_GATE_MIN_MODE_ROWS", DEFAULT_CURRICULUM_MIN_MODE_ROWS))


def source_trace_curriculum_positive_rows(source_payload: dict[str, Any]) -> int:
    if source_payload.get("kind") != "stage5_capability_ladder_trace_collection":
        return 0
    curriculum = source_payload.get("curriculum") if isinstance(source_payload.get("curriculum"), dict) else {}
    counts = curriculum.get("counts") if isinstance(curriculum.get("counts"), dict) else {}
    return int(counts.get("positive_sft_rows") or 0)


def curriculum_sft_validation_no_go_reason(source_payload: dict[str, Any]) -> str | None:
    if source_payload.get("kind") != "stage5_curriculum_sft":
        return None
    checks = source_payload.get("validation_checks")
    if isinstance(checks, dict):
        status = str(checks.get("status") or "")
        if status and status != "validation_sane":
            return (
                f"Curriculum SFT summary reports validation status {status!r} with "
                f"issues {checks.get('issues') or []!r}; inspect locally before paid routing diagnostics."
            )
        depth_gradient = checks.get("depth_gradient")
        if isinstance(depth_gradient, dict):
            if depth_gradient.get("available") is False:
                return (
                    f"Curriculum SFT summary is missing direct/deep depth-gradient metrics {depth_gradient!r}; "
                    "inspect locally before paid routing diagnostics."
                )
            if depth_gradient.get("observed") is False:
                return (
                    f"Curriculum SFT summary did not observe the required direct-vs-deep depth gradient "
                    f"{depth_gradient!r}; inspect locally before paid routing diagnostics."
                )
        elif checks:
            return (
                "Curriculum SFT summary has validation checks but no depth-gradient diagnostic; "
                "inspect locally before paid routing diagnostics."
            )
    phase1_val = source_payload.get("phase1_val")
    if not isinstance(phase1_val, dict) or not phase1_val:
        return "Curriculum SFT summary is missing phase1_val metrics; inspect locally before paid routing diagnostics."
    nonfinite = [
        key
        for key, value in phase1_val.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    ]
    if nonfinite:
        return (
            f"Curriculum SFT validation has non-finite metrics {nonfinite}; "
            "inspect locally before paid routing diagnostics."
        )
    mean_loops = phase1_val.get("mean_expected_loops")
    if isinstance(mean_loops, (int, float)) and float(mean_loops) < CURRICULUM_SFT_MIN_MEAN_EXPECTED_LOOPS:
        return (
            f"Curriculum SFT mean_expected_loops={float(mean_loops):.4g} is below "
            f"{CURRICULUM_SFT_MIN_MEAN_EXPECTED_LOOPS:.4g}; inspect loop collapse before paid routing diagnostics."
        )
    return None


def source_has_calibration_warning(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", ""))
    if payload.get("decision") == "stop_for_calibration_repair":
        return True
    if status.endswith("_calibration_warning"):
        return True
    best = payload.get("best_arm") or {}
    if isinstance(best, dict):
        comparison = ((best.get("best_checkpoint") or {}).get("comparison_to_base") or {})
        if comparison.get("calibration_ok") is False:
            return True
    return False


def source_is_clean_full_confirmation_proxy(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", ""))
    if not bool(payload.get("passed", False)):
        return False
    if status in {"proxy_lift", "proxy_matches_base"}:
        return True
    if status in {"repair_proxy_lift", "repair_proxy_matches_base"}:
        alignment = payload.get("proxy_alignment")
        if isinstance(alignment, dict) and alignment.get("ok") is False:
            return False
        return True
    return False


def promoted_stage4_opus_sources(source_payload: dict[str, Any]) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for item in source_payload.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") != "promote_to_small_train_mix":
            continue
        if str(item.get("key") or "") not in STAGE4_OPUS_APPROVED_SOURCE_KEYS:
            continue
        if item.get("avoid_for_now"):
            continue
        if int(item.get("converted_rows") or 0) <= 0:
            continue
        if float(item.get("conversion_rate") or 0.0) < 0.5:
            continue
        promoted.append(item)
    return promoted


def infer_stage5_run_id(path: str | Path) -> str | None:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part == "stage5" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def checkpoint_from_payload(payload: dict[str, Any]) -> str | None:
    """Find the checkpoint the guarded next action is expected to consume."""

    direct = payload.get("selected_checkpoint") or payload.get("checkpoint") or payload.get("phase1_checkpoint")
    if direct:
        return str(direct)

    for key_path in [
        ("best_checkpoint", "checkpoint"),
        ("best_arm", "best_checkpoint", "checkpoint"),
        ("balanced_assessment", "best_checkpoint", "checkpoint"),
    ]:
        current: Any = payload
        for key in key_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current:
            return str(current)

    source_summary = payload.get("source_summary")
    if source_summary:
        source_path = Path(str(source_summary))
        source_path = source_path if source_path.is_absolute() else ROOT / source_path
        if source_path.exists():
            try:
                nested = read_json(source_path)
            except Exception:
                nested = {}
            nested_checkpoint = checkpoint_from_payload(nested)
            if nested_checkpoint:
                return nested_checkpoint

    return None


def checkpoint_availability_for_path(
    checkpoint: str | Path | None,
    *,
    run_id_hint: str | None = None,
    missing_reason: str = "No selected checkpoint was found in the source summary.",
) -> dict[str, Any]:
    if not checkpoint:
        return {
            "checkpoint": None,
            "available": False,
            "exists": False,
            "drive_candidate_exists": False,
            "reason": missing_reason,
        }

    checkpoint_path = Path(checkpoint)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path
    run_id = run_id_hint or infer_stage5_run_id(checkpoint_path)
    exists = checkpoint_path.exists()
    candidates: list[Path] = []
    existing_candidates: list[Path] = []
    if not exists and run_id:
        candidates = candidate_drive_checkpoints(run_id, checkpoint_path.name)
        existing_candidates = [path for path in candidates if path.exists()]

    return {
        "checkpoint": checkpoint_path_for_cli(checkpoint_path).replace("\\", "/"),
        "available": bool(exists or existing_candidates),
        "exists": exists,
        "run_id": run_id,
        "drive_candidates_checked": min(len(candidates), 12),
        "drive_candidate_exists": bool(existing_candidates),
        "first_existing_drive_candidate": str(existing_candidates[0]) if existing_candidates else None,
        "first_drive_candidates": [str(path) for path in candidates[:12]],
    }


def checkpoint_availability(payload: dict[str, Any]) -> dict[str, Any]:
    return checkpoint_availability_for_path(checkpoint_from_payload(payload))


def checkpoint_from_summary_reference(value: str | Path | None) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return checkpoint_from_payload(payload)


def default_recovered_checkpoint_availability() -> dict[str, Any]:
    """Preflight the default recovered deterministic checkpoint."""

    checkpoint = os.environ.get("STAGE5_RECOVERED_PHASE1_CHECKPOINT", DEFAULT_CHECKPOINT_REL)
    run_id = os.environ.get("STAGE5_RECOVERED_PHASE1_RUN_ID", DEFAULT_RECOVERED_RUN_ID)
    status = checkpoint_availability_for_path(
        checkpoint,
        run_id_hint=run_id,
        missing_reason="Routing diagnostic/repair requires the recovered deterministic Phase 1 checkpoint.",
    )
    status["reason"] = (
        "Routing diagnostic/repair requires the recovered deterministic Phase 1 checkpoint. "
        "Run Drive/checkpoint preflight before attaching a paid GPU if this is unavailable."
    )
    return status


def routing_repair_checkpoint_availability(source_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preflight the checkpoint the routing-repair child runner will consume."""

    checkpoint = None
    source = "default_recovered"
    if source_payload:
        checkpoint = checkpoint_from_summary_reference(source_payload.get("benchmark_summary"))
        if checkpoint:
            source = "benchmark_summary"
        else:
            checkpoint = checkpoint_from_payload(source_payload)
            if checkpoint:
                source = "source_payload"
    if not checkpoint:
        return default_recovered_checkpoint_availability()

    status = checkpoint_availability_for_path(
        checkpoint,
        missing_reason="Routing repair requires the benchmark checkpoint selected by the routing diagnostic.",
    )
    status["source"] = source
    status["reason"] = (
        "Routing repair will delegate to the ARC-mix child runner using this selected checkpoint. "
        "It must be local or visible in the Drive artifact backup before attaching a paid GPU."
    )
    return status


def programmatic_depth_checkpoint_availability(source_payload: dict[str, Any]) -> dict[str, Any]:
    """Preflight the checkpoint the constructed direct/deep repair will resume from."""

    explicit = os.environ.get("STAGE5_PROGRAMMATIC_RESUME_CHECKPOINT", "").strip()
    checkpoint = explicit or checkpoint_from_payload(source_payload)
    status = checkpoint_availability_for_path(
        checkpoint,
        missing_reason=(
            "Programmatic direct/deep repair requires STAGE5_PROGRAMMATIC_RESUME_CHECKPOINT "
            "or a checkpoint in the source summary."
        ),
    )
    status["reason"] = (
        "Programmatic direct/deep repair resumes from this selected checkpoint. "
        "It must be local or visible in the Drive artifact backup before attaching a paid GPU."
    )
    return status


def routing_repair_profile_preflight(source_payload: dict[str, Any]) -> dict[str, Any]:
    status = str(source_payload.get("status") or "")
    if status not in {"needs_direct_halting_repair", "needs_deep_narrow_recovery"}:
        return {
            "checked": False,
            "status": status,
            "reason": "Source status is not a routing-repair status.",
        }
    from colab.run_stage5_routing_repair import repair_profile

    profile = repair_profile(status)
    return {
        "checked": True,
        "status": status,
        "repair_mode": profile.get("repair_mode"),
        "expected_arc_eval_config": profile.get("STAGE5_ARC_MIX_EVAL_CONFIG"),
        "arms": profile.get("STAGE5_ARC_MIX_ARMS"),
        "arc_easy_target_loop": profile.get("STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP"),
        "arc_challenge_target_loop": profile.get("STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP"),
    }


def curriculum_sft_checkpoint_availability() -> dict[str, Any]:
    resume_from = os.environ.get("STAGE5_CURRICULUM_RESUME_FROM", "").strip()
    if not resume_from:
        return {
            "checkpoint": None,
            "available": True,
            "exists": False,
            "drive_candidate_exists": False,
            "reason": (
                "Generated-curriculum SFT starts from the base model because "
                "STAGE5_CURRICULUM_RESUME_FROM is not set."
            ),
        }
    status = checkpoint_availability_for_path(
        resume_from,
        missing_reason=(
            "Generated-curriculum SFT was configured with "
            "STAGE5_CURRICULUM_RESUME_FROM, but that checkpoint was not found."
        ),
    )
    status["reason"] = (
        "Generated-curriculum SFT will resume from STAGE5_CURRICULUM_RESUME_FROM; "
        "the checkpoint must be local or visible in the Drive artifact backup before using a paid GPU."
    )
    return status


def curriculum_input_backup_root() -> Path:
    return Path(
        os.environ.get(
            "STAGE5_CURRICULUM_INPUT_BACKUP_DIR",
            "/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs",
        )
    )


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def curriculum_work_dir_backup_candidates(work_dir: Path | None) -> list[Path]:
    if work_dir is None:
        return []
    root = curriculum_input_backup_root()
    candidates = [root / work_dir.name]
    try:
        relative = work_dir.relative_to(ROOT)
    except ValueError:
        relative = Path(work_dir.name)
    candidates.append(root / relative)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def curriculum_sft_input_availability(source_payload: dict[str, Any]) -> dict[str, Any]:
    curriculum = source_payload.get("curriculum") if isinstance(source_payload.get("curriculum"), dict) else {}
    nested_gate = source_payload.get("gate") if isinstance(source_payload.get("gate"), dict) else {}
    work_dir = resolve_repo_path(
        os.environ.get("STAGE5_CURRICULUM_WORK_DIR") or source_payload.get("work_dir") or curriculum.get("work_dir")
    )
    summary_json = resolve_repo_path(
        os.environ.get("STAGE5_CURRICULUM_SUMMARY_JSON")
        or source_payload.get("summary_json")
        or curriculum.get("summary_json")
    )
    source_artifacts = source_payload.get("artifacts") if isinstance(source_payload.get("artifacts"), dict) else {}
    gate_artifacts = nested_gate.get("artifacts") if isinstance(nested_gate.get("artifacts"), dict) else {}
    positive_sft = resolve_repo_path(source_artifacts.get("positive_sft") or gate_artifacts.get("positive_sft"))

    # Older summaries may not include these fields. In that case let the runner
    # perform its own validation instead of blocking a valid legacy handoff.
    if work_dir is None and summary_json is None and positive_sft is None:
        return {
            "available": True,
            "reason": "Curriculum gate summary does not include input paths; runner will validate local artifacts.",
            "work_dir": None,
            "summary_json": None,
            "positive_sft": None,
        }

    local_available = bool(
        (work_dir is None or work_dir.exists())
        and (summary_json is None or summary_json.exists())
        and (positive_sft is None or positive_sft.exists())
    )
    if local_available:
        return {
            "available": True,
            "reason": "Curriculum input artifacts are present locally.",
            "work_dir": None if work_dir is None else path_for_cli(work_dir),
            "summary_json": None if summary_json is None else path_for_cli(summary_json),
            "positive_sft": None if positive_sft is None else path_for_cli(positive_sft),
            "local_available": True,
        }

    candidates = curriculum_work_dir_backup_candidates(work_dir)
    existing = [
        candidate
        for candidate in candidates
        if (candidate / "summary.json").exists() and (candidate / "positive_sft.jsonl").exists()
    ]
    if existing:
        return {
            "available": True,
            "reason": "Curriculum input artifacts are missing locally but visible in the Drive curriculum backup.",
            "work_dir": None if work_dir is None else path_for_cli(work_dir),
            "summary_json": None if summary_json is None else path_for_cli(summary_json),
            "positive_sft": None if positive_sft is None else path_for_cli(positive_sft),
            "local_available": False,
            "drive_candidate_exists": True,
            "first_existing_drive_candidate": str(existing[0]),
            "first_drive_candidates": [str(path) for path in candidates[:12]],
        }

    return {
        "available": False,
        "reason": (
            "Curriculum SFT input artifacts are not present locally and no Drive curriculum backup "
            "with summary.json and positive_sft.jsonl is visible."
        ),
        "work_dir": None if work_dir is None else path_for_cli(work_dir),
        "summary_json": None if summary_json is None else path_for_cli(summary_json),
        "positive_sft": None if positive_sft is None else path_for_cli(positive_sft),
        "local_available": False,
        "drive_candidate_exists": False,
        "first_drive_candidates": [str(path) for path in candidates[:12]],
        "curriculum_input_backup_root": str(curriculum_input_backup_root()),
    }


def curriculum_sft_preflight(source_payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint = curriculum_sft_checkpoint_availability()
    inputs = curriculum_sft_input_availability(source_payload)
    return {
        **checkpoint,
        "available": bool(checkpoint.get("available") and inputs.get("available")),
        "checkpoint_preflight": checkpoint,
        "input_preflight": inputs,
    }


def apply_checkpoint_guard(
    decision: dict[str, Any],
    *,
    source_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if decision.get("spend_class") == "bounded_routing_diagnostic":
        checkpoint = default_recovered_checkpoint_availability()
    elif decision.get("spend_class") == "bounded_routing_repair":
        checkpoint = routing_repair_checkpoint_availability(source_payload)
    elif decision.get("spend_class") == "bounded_programmatic_depth_repair":
        checkpoint = programmatic_depth_checkpoint_availability(source_payload)
    elif decision.get("spend_class") == "bounded_stage4_opus_finetune":
        checkpoint = {
            "checkpoint": None,
            "available": True,
            "exists": False,
            "drive_candidate_exists": False,
            "reason": "Stage 4 Opus fine-tune starts from the base model and does not require a recovered checkpoint preflight.",
        }
    elif decision.get("spend_class") == "bounded_curriculum_sft":
        checkpoint = curriculum_sft_preflight(source_payload)
    elif decision.get("checkpoint"):
        checkpoint = checkpoint_availability_for_path(str(decision["checkpoint"]))
    elif decision.get("spend_class") in {"bounded_dense_arc_sft"}:
        checkpoint = {
            "checkpoint": None,
            "available": True,
            "exists": False,
            "drive_candidate_exists": False,
            "reason": "Dense ARC SFT is the standard-model control and starts from the base model.",
        }
    else:
        checkpoint = checkpoint_availability(source_payload)
    if not decision.get("go"):
        return decision, checkpoint
    if checkpoint.get("available"):
        return decision, checkpoint
    if decision.get("spend_class") == "bounded_curriculum_sft" and not (
        checkpoint.get("input_preflight") or {}
    ).get("available"):
        guarded = {
            "go": False,
            "status": "curriculum_input_missing_no_go",
            "spend_class": "none",
            "reason": (
                "The planner selected generated-curriculum SFT, but the curriculum work dir "
                "is not present locally and no Drive backup candidate is visible. Run or restore "
                "the CPU/API curriculum artifact pipeline before attaching a paid GPU."
            ),
            "prior_decision": decision,
        }
        return guarded, checkpoint
    if decision.get("spend_class") in {
        "bounded_routing_diagnostic",
        "bounded_routing_repair",
        "bounded_programmatic_depth_repair",
    }:
        guarded = {
            "go": False,
            "status": "routing_checkpoint_missing_no_go",
            "spend_class": "none",
            "reason": (
                "The planner selected a routing diagnostic/repair, but the recovered "
                "Phase 1 checkpoint is not present locally and no mounted-Drive backup "
                "candidate is visible. Mount/authorize Drive on a cheap runtime first."
            ),
            "prior_decision": decision,
        }
        return guarded, checkpoint
    guarded = {
        "go": False,
        "status": "checkpoint_missing_no_go",
        "spend_class": "none",
        "reason": (
            "The planner selected a paid-GPU action, but the checkpoint is not "
            "present locally and no Drive backup candidate is visible."
        ),
        "prior_decision": decision,
    }
    return guarded, checkpoint


def classify_action(
    action: dict[str, Any] | None,
    *,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    if not action:
        return {
            "go": False,
            "status": "no_planner_action",
            "spend_class": "none",
            "reason": "Planner returned no action.",
        }

    command = str(action.get("command", ""))
    script = command_script(command)
    source_status = str(source_payload.get("status", "unknown"))
    source_kind_label = source_payload_kind(source_payload)

    if script == "colab/run_stage5_routing_diagnostic.py":
        validation_reason = curriculum_sft_validation_no_go_reason(source_payload)
        if validation_reason:
            return {
                "go": False,
                "status": "curriculum_sft_validation_no_go",
                "spend_class": "none",
                "reason": validation_reason,
            }
        return {
            "go": True,
            "status": "go_routing_diagnostic",
            "spend_class": "bounded_routing_diagnostic",
            "reason": "Planner recommends a bounded direct/deep routing diagnostic before further training.",
        }

    if script == "colab/run_stage5_routing_repair.py":
        routing_status = str(source_payload.get("status", ""))
        if routing_status in {"needs_direct_halting_repair", "needs_deep_narrow_recovery"}:
            return {
                "go": True,
                "status": "go_routing_repair",
                "spend_class": "bounded_routing_repair",
                "reason": "Routing diagnostic selected one bounded deterministic Phase 1 repair.",
            }
        return {
            "go": False,
            "status": "routing_repair_blocked",
            "spend_class": "none",
            "reason": f"Routing repair requires a repair status, got {routing_status!r}.",
        }

    if script == "colab/run_stage5_programmatic_depth_repair.py":
        routing_status = str(source_payload.get("status", ""))
        if routing_status in {
            "needs_direct_halting_repair",
            "needs_deep_narrow_recovery",
            "repair_no_proxy_lift",
            "repair_proxy_lift_calibration_warning",
            "repair_proxy_matches_base_calibration_warning",
        }:
            return {
                "go": True,
                "status": "go_programmatic_depth_repair",
                "spend_class": "bounded_programmatic_depth_repair",
                "reason": (
                    "Routing/repair evidence indicates deterministic depth/direct repair is still needed; "
                    "one bounded constructed-curriculum repair is allowed."
                ),
            }
        return {
            "go": False,
            "status": "programmatic_depth_repair_blocked",
            "spend_class": "none",
            "reason": f"Programmatic depth repair requires a repair status, got {routing_status!r}.",
        }

    if script == "colab/run_stage5_direct_preservation_probe.py":
        direct_preservation_source = (
            source_kind_label == "arc_mix_answer_prior_diagnosis"
            and source_payload.get("status") == "direct_answer_prior_not_preserved"
        ) or (
            source_kind_label == "mcq_debias_diagnostic"
            and source_payload.get("status") == "content_degradation_persists"
        )
        if direct_preservation_source:
            source_summary = str(source_payload.get("nested_source_summary") or source_payload.get("source_summary") or "").strip()
            checkpoint = None
            if source_summary:
                try:
                    nested = read_json(resolve_path(source_summary))
                    checkpoint = str(nested.get("resume_checkpoint") or "").replace("\\", "/")
                except Exception:
                    checkpoint = None
            return go_paid_gpu_action(
                status="go_direct_preservation_probe",
                spend_class="bounded_direct_preservation_probe",
                checkpoint=checkpoint,
                reason=(
                    "Debiased evidence still shows the direct route is not preserving base-confident examples; "
                    "one bounded max_loops=1 direct-preservation probe is allowed."
                ),
            )
        return {
            "go": False,
            "status": "direct_preservation_probe_blocked",
            "spend_class": "none",
            "reason": (
                "Direct preservation probe requires either source kind arc_mix_answer_prior_diagnosis with "
                "status direct_answer_prior_not_preserved, or source kind mcq_debias_diagnostic with "
                "status content_degradation_persists."
            ),
        }

    if script == "colab/run_stage5_mcq_debias_diagnostic.py":
        initial_debias_source = (
            source_kind_label == "arc_mix_answer_prior_diagnosis"
            and source_payload.get("status") == "direct_answer_prior_not_preserved"
        )
        confirmation_debias_source = (
            source_kind_label == "mcq_debias_diagnostic"
            and source_payload.get("status") == "selection_bias_likely"
        )
        if initial_debias_source or confirmation_debias_source:
            source_summary = str(source_payload.get("nested_source_summary") or source_payload.get("source_summary") or "").strip()
            checkpoint = None
            if source_summary:
                try:
                    nested = read_json(resolve_path(source_summary))
                    checkpoint = str(nested.get("resume_checkpoint") or "").replace("\\", "/")
                except Exception:
                    checkpoint = None
            return go_paid_gpu_action(
                status="go_mcq_debias_diagnostic",
                spend_class="bounded_mcq_debias_diagnostic",
                checkpoint=checkpoint,
                reason=(
                    "MCQ option-label selection bias is unresolved; one bounded no-training "
                    "label/content/permutation debias diagnostic is allowed before direct-preservation training."
                ),
            )
        return {
            "go": False,
            "status": "mcq_debias_diagnostic_blocked",
            "spend_class": "none",
            "reason": (
                "MCQ debias diagnostic requires either source kind arc_mix_answer_prior_diagnosis with "
                "status direct_answer_prior_not_preserved, or source kind mcq_debias_diagnostic with "
                "status selection_bias_likely."
            ),
        }

    if script == "colab/run_stage4_opus_finetune.py":
        promoted = promoted_stage4_opus_sources(source_payload)
        if promoted:
            return {
                "go": True,
                "status": "go_stage4_opus_finetune",
                "spend_class": "bounded_stage4_opus_finetune",
                "reason": "Dataset audit promoted a compatible Opus-style trace source; one bounded Stage 4 fine-tune is allowed.",
            }
        return {
            "go": False,
            "status": "stage4_opus_finetune_blocked",
            "spend_class": "none",
            "reason": (
                "Stage 4 Opus fine-tune requires a dataset audit with a promoted, approved "
                "Opus recovery source such as opus47_sft or opus47_raw."
            ),
        }

    if script == "colab/run_stage5_curriculum_sft.py":
        trace_collection_gate_ready = (
            source_payload.get("kind") == "stage5_capability_ladder_trace_collection"
            and source_payload.get("status") == "trace_curriculum_gate_ready"
            and isinstance(source_payload.get("gate"), dict)
            and (source_payload.get("gate") or {}).get("go") is True
        )
        standalone_gate_ready = source_payload.get("kind") == "curriculum_sft_gate" and source_payload.get("go") is True
        if standalone_gate_ready or trace_collection_gate_ready:
            env = command_env_assignments(command)
            if trace_collection_gate_ready:
                min_trace_rows = int(
                    env.get("STAGE5_TRACE_CURRICULUM_MIN_SFT_ROWS")
                    or os.environ.get("STAGE5_TRACE_CURRICULUM_MIN_SFT_ROWS")
                    or DEFAULT_TRACE_CURRICULUM_MIN_SFT_ROWS
                )
                observed_trace_rows = source_trace_curriculum_positive_rows(source_payload)
                if observed_trace_rows < min_trace_rows:
                    return {
                        "go": False,
                        "status": "curriculum_sft_too_few_trace_rows",
                        "spend_class": "none",
                        "reason": (
                            "Traced capability-ladder SFT requires enough answer-verified rows before paid GPU "
                            f"training. Expected at least {min_trace_rows}, got {observed_trace_rows}. "
                            "Collect more provider responses or deliberately lower STAGE5_TRACE_CURRICULUM_MIN_SFT_ROWS "
                            "for a tiny smoke run."
                        ),
                    }
            expected_min_mode_rows = source_curriculum_min_mode_rows(source_payload)
            actual_min_mode_rows = normalize_min_mode_rows(env.get("STAGE5_CURRICULUM_MIN_MODE_ROWS", ""))
            if expected_min_mode_rows and actual_min_mode_rows != expected_min_mode_rows:
                return {
                    "go": False,
                    "status": "curriculum_sft_mode_gate_mismatch",
                    "spend_class": "none",
                    "reason": (
                        "Generated curriculum SFT requires an explicit mode-coverage gate in "
                        "STAGE5_CURRICULUM_MIN_MODE_ROWS before paid GPU training. "
                        f"Expected {expected_min_mode_rows!r}, got {env.get('STAGE5_CURRICULUM_MIN_MODE_ROWS')!r}."
                    ),
                }
            return {
                "go": True,
                "status": "go_curriculum_sft",
                "spend_class": "bounded_curriculum_sft",
                "reason": (
                    "A generated curriculum shard passed the strict SFT gate; "
                    "one bounded deterministic recurrent Phase 1 SFT run is allowed."
                ),
            }
        return {
            "go": False,
            "status": "curriculum_sft_blocked",
            "spend_class": "none",
            "reason": (
                "Generated curriculum SFT requires a source summary with kind=curriculum_sft_gate and go=true, "
                "or a stage5_capability_ladder_trace_collection summary with status=trace_curriculum_gate_ready "
                "and gate.go=true."
            ),
        }

    if source_has_calibration_warning(source_payload):
        return {
            "go": False,
            "status": "calibration_warning_no_go",
            "spend_class": "none",
            "reason": "Source summary reports calibration warning; inspect locally before using A100.",
        }

    if script == "colab/run_stage5_arc_agi_candidate_gate.py":
        if source_kind_label == "bootstrap":
            return go_paid_gpu_action(
                status="go_arc_agi_candidate_gate",
                spend_class="bounded_arc_agi_candidate_gate",
                checkpoint=default_phase1_checkpoint(command),
                reason="No Stage 5 summary exists yet; one bounded ARC-AGI candidate-source gate is the planned bootstrap.",
            )
        return {
            "go": False,
            "status": "candidate_gate_blocked",
            "spend_class": "none",
            "reason": f"ARC-AGI candidate gate is only allowed as the bootstrap paid-GPU action, got source kind {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_trace_sft_gate.py":
        if source_kind_label in {"candidate_gate", "recovery_particle_gate"}:
            return go_paid_gpu_action(
                status="go_trace_sft_gate",
                spend_class="bounded_trace_sft_gate",
                checkpoint=default_phase1_checkpoint(command),
                reason="Planner selected a bounded trace-target SFT gate from candidate or recovery evidence.",
            )
        return {
            "go": False,
            "status": "trace_sft_gate_blocked",
            "spend_class": "none",
            "reason": f"Trace SFT gate requires candidate/recovery-particle evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_distill_sft_gate.py":
        if source_kind_label == "trace_sft_gate":
            return go_paid_gpu_action(
                status="go_distill_sft_gate",
                spend_class="bounded_distill_sft_gate",
                checkpoint=default_phase1_checkpoint(command),
                reason="Trace-SFT gate selected a recipe arm; one bounded distillation gate is allowed.",
            )
        return {
            "go": False,
            "status": "distill_sft_gate_blocked",
            "spend_class": "none",
            "reason": f"Distillation gate requires trace-SFT gate evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_dense_sft.py":
        if source_kind_label in {"candidate_gate", "trace_sft_gate", "distill_sft_gate", "recipe_control_assessment"}:
            return go_paid_gpu_action(
                status="go_dense_arc_sft_control",
                spend_class="bounded_dense_arc_sft",
                reason="Planner selected a bounded dense same-recipe control needed before architecture-lift claims.",
            )
        return {
            "go": False,
            "status": "dense_arc_sft_blocked",
            "spend_class": "none",
            "reason": f"Dense ARC SFT control requires gate/control evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_sft.py":
        if source_kind_label == "dense_sft_control":
            return go_paid_gpu_action(
                status="go_matched_recurrent_arc_sft",
                spend_class="bounded_matched_recurrent_arc_sft",
                checkpoint=default_phase1_checkpoint(command),
                reason="Dense control exists; run the matched recurrent recipe arm under the same ARC row recipe.",
            )
        return {
            "go": False,
            "status": "matched_recurrent_arc_sft_blocked",
            "spend_class": "none",
            "reason": f"Matched recurrent ARC SFT requires a dense-control source, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_candidate_distill_gate.py":
        if source_kind_label in {"benchmark", "autopilot", "followup", "selector_rescore"}:
            return go_paid_gpu_action(
                status="go_candidate_distill_gate",
                spend_class="bounded_candidate_distill_gate",
                checkpoint=recovered_checkpoint_from_command(command) or default_phase1_checkpoint(command),
                reason="Planner selected a bounded candidate-distillation diagnostic/gate after benchmark or selector evidence.",
            )
        return {
            "go": False,
            "status": "candidate_distill_gate_blocked",
            "spend_class": "none",
            "reason": f"Candidate distillation gate requires benchmark/autopilot/selector evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_curriculum_particle_autopilot.py":
        if source_kind_label in {"benchmark", "autopilot", "followup"}:
            return go_paid_gpu_action(
                status="go_curriculum_particle_autopilot",
                spend_class="bounded_curriculum_particle_autopilot",
                checkpoint=default_phase1_checkpoint(command),
                reason="Planner selected a bounded deterministic curriculum/particle autopilot from benchmark evidence.",
            )
        return {
            "go": False,
            "status": "curriculum_particle_autopilot_blocked",
            "spend_class": "none",
            "reason": f"Curriculum-particle autopilot requires benchmark/autopilot evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_autopilot_followup.py":
        if source_kind_label in {"autopilot", "benchmark", "followup"}:
            return go_paid_gpu_action(
                status="go_autopilot_followup",
                spend_class="bounded_autopilot_followup",
                checkpoint=recovered_checkpoint_from_command(command) or checkpoint_from_payload(source_payload),
                reason="Planner selected a bounded follow-up evaluation from an autopilot/benchmark summary.",
            )
        return {
            "go": False,
            "status": "autopilot_followup_blocked",
            "spend_class": "none",
            "reason": f"Autopilot follow-up requires autopilot/benchmark evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_recovered_benchmark.py":
        if source_kind_label in {"recovery_particle_gate", "autopilot", "selector_rescore", "followup"}:
            return go_paid_gpu_action(
                status="go_recovered_arc_benchmark",
                spend_class="bounded_recovered_arc_benchmark",
                checkpoint=recovered_checkpoint_from_command(command) or checkpoint_from_payload(source_payload),
                reason="Planner selected a bounded recovered-vs-base ARC benchmark from recovered checkpoint evidence.",
            )
        return {
            "go": False,
            "status": "recovered_arc_benchmark_blocked",
            "spend_class": "none",
            "reason": f"Recovered ARC benchmark requires recovery/autopilot/selector evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_recovery_particle_gate.py":
        if source_kind_label == "recovery_particle_gate":
            return go_paid_gpu_action(
                status="go_recovery_particle_gate",
                spend_class="bounded_recovery_particle_gate",
                checkpoint=default_phase1_checkpoint(command),
                reason="Planner selected a bounded replicated recovery/particle gate from prior recovery-particle evidence.",
            )
        return {
            "go": False,
            "status": "recovery_particle_gate_blocked",
            "spend_class": "none",
            "reason": f"Recovery-particle gate requires recovery-particle source evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_arc_agi_tta_sweep.py":
        if source_kind_label in {"autopilot", "followup", "benchmark", "selector_rescore"}:
            return go_paid_gpu_action(
                status="go_arc_tta_sweep",
                spend_class="bounded_arc_tta_sweep",
                checkpoint=recovered_checkpoint_from_command(command) or checkpoint_from_payload(source_payload),
                reason="Planner selected a bounded ARC TTA/selector sweep from existing benchmark evidence.",
            )
        return {
            "go": False,
            "status": "arc_tta_sweep_blocked",
            "spend_class": "none",
            "reason": f"ARC TTA sweep requires benchmark/autopilot/selector evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_phase1_recovery_ladder.py":
        if source_kind_label == "stage4_opus_finetune":
            return go_paid_gpu_action(
                status="go_phase1_recovery_ladder",
                spend_class="bounded_phase1_recovery_ladder",
                checkpoint=default_phase1_checkpoint(command),
                reason="Planner selected bounded deterministic recurrent recovery after Stage 4 Opus fine-tune evidence.",
            )
        return {
            "go": False,
            "status": "phase1_recovery_ladder_blocked",
            "spend_class": "none",
            "reason": f"Phase1 recovery ladder requires Stage 4 Opus evidence, got {source_kind_label!r}.",
        }

    if script in {
        "colab/run_stage5_recovered_phase1_arc_gate.py",
        "colab/run_stage5_recovered_phase1_particle_arc_gate.py",
        "colab/run_stage5_recovered_phase2_smoke.py",
    }:
        if source_kind_label in {"stage4_opus_finetune", "recurrent_sft", "benchmark", "autopilot"}:
            return go_paid_gpu_action(
                status="go_recovered_stage5_gate",
                spend_class="bounded_recovered_stage5_gate",
                checkpoint=recovered_checkpoint_from_command(command) or checkpoint_from_payload(source_payload),
                reason="Planner selected a bounded recovered-checkpoint gate from prior training evidence.",
            )
        return {
            "go": False,
            "status": "recovered_stage5_gate_blocked",
            "spend_class": "none",
            "reason": f"Recovered checkpoint gates require training/benchmark evidence, got {source_kind_label!r}.",
        }

    if script == "colab/run_stage5_balanced_arc_mix_gate.py":
        env = command_env_assignments(command)
        return go_paid_gpu_action(
            status="go_bounded_proxy",
            spend_class="single_arc_mix_proxy",
            checkpoint=checkpoint_from_summary_reference(env.get("STAGE5_ARC_MIX_SOURCE_SUMMARY"))
            or checkpoint_from_payload(source_payload),
            reason="Planner recommends exactly one bounded competence-recovery proxy.",
        )

    if script == "colab/run_stage5_recovery_full_assessment.py":
        if source_is_clean_full_confirmation_proxy(source_payload):
            return {
                "go": True,
                "status": "go_full_confirmation",
                "spend_class": "single_full_balanced_assessment",
                "reason": "A clean proxy passed; one full balanced confirmation is justified.",
            }
        return {
            "go": False,
            "status": "full_assessment_blocked",
            "spend_class": "none",
            "reason": "Full assessment requires a passed, non-warning ARC-mix proxy summary.",
        }

    if script == "colab/run_stage5_benchmark_suite.py":
        benchmark_budget = benchmark_suite_budget_preflight(command)
        if not benchmark_budget["go"]:
            return {
                "go": False,
                "status": "benchmark_suite_limit_no_go",
                "spend_class": "none",
                "reason": benchmark_budget["reason"],
                "benchmark_budget": benchmark_budget,
            }
        return {
            "go": True,
            "status": "go_broader_benchmark",
            "spend_class": "bounded_benchmark_suite",
            "reason": "Planner found a nonnegative balanced checkpoint and recommends broader benchmarks.",
            "benchmark_budget": benchmark_budget,
        }

    if script == "colab/run_stage5_capability_ladder_mcq_probe.py":
        probe_budget = capability_ladder_probe_budget_preflight(command)
        if not probe_budget["go"]:
            return {
                "go": False,
                "status": "capability_ladder_probe_limit_no_go",
                "spend_class": "none",
                "reason": probe_budget["reason"],
                "capability_ladder_probe_budget": probe_budget,
            }
        return {
            "go": True,
            "status": "go_capability_ladder_mcq_probe",
            "spend_class": "bounded_capability_ladder_mcq_probe",
            "reason": (
                "Planner/user selected a bounded no-training model-scale MCQ scoring probe "
                "to build depth-targeted capability-ladder rows."
            ),
            "capability_ladder_probe_budget": probe_budget,
        }

    return {
        "go": False,
        "status": "no_gpu_action",
        "spend_class": "none",
        "reason": "Planner action is inspection, documentation, or another non-GPU/local step.",
    }


def build_payload(source_summary: Path) -> dict[str, Any]:
    source_payload = read_json(source_summary)
    actions = plan_next_actions(source_payload, source_summary=source_summary)
    action = actions[0] if actions else None
    action_env = command_env_assignments(str((action or {}).get("command", "")))
    with scoped_environ(action_env):
        decision = classify_action(action, source_payload=source_payload)
        decision, checkpoint = apply_checkpoint_guard(decision, source_payload=source_payload)
    routing_profile = routing_repair_profile_preflight(source_payload)
    return {
        "run_id": RUN_ID,
        "kind": "stage5_a100_go_no_go",
        "source_summary": path_for_cli(source_summary),
        "source_kind": source_kind(source_payload),
        "source_status": source_payload.get("status"),
        "decision": decision,
        "checkpoint_preflight": checkpoint,
        "routing_repair_profile": routing_profile,
        "recommended_action": action,
        "all_actions": actions,
    }


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    action = payload.get("recommended_action") or {}
    decision = payload["decision"]
    checkpoint = payload.get("checkpoint_preflight") or {}
    input_preflight = checkpoint.get("input_preflight") or {}
    routing_profile = payload.get("routing_repair_profile") or {}
    lines = [
        f"# Stage 5 A100 Go/No-Go - {payload['run_id']}",
        "",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Source kind: `{payload['source_kind']}`",
        f"- Source status: `{payload['source_status']}`",
        f"- Decision: `{decision['status']}`",
        f"- Go: `{decision['go']}`",
        f"- Spend class: `{decision['spend_class']}`",
        f"- Reason: {decision['reason']}",
        f"- Checkpoint: `{checkpoint.get('checkpoint')}`",
        f"- Checkpoint available: `{checkpoint.get('available')}`",
        f"- Checkpoint exists locally: `{checkpoint.get('exists')}`",
        f"- Drive candidate visible: `{checkpoint.get('drive_candidate_exists')}`",
        f"- Curriculum input available: `{input_preflight.get('available')}`",
        f"- Curriculum input local: `{input_preflight.get('local_available')}`",
        f"- Curriculum input Drive candidate visible: `{input_preflight.get('drive_candidate_exists')}`",
        f"- Routing repair mode: `{routing_profile.get('repair_mode')}`",
        f"- Routing repair proxy eval: `{routing_profile.get('expected_arc_eval_config')}`",
        f"- Routing repair arms: `{routing_profile.get('arms')}`",
        "",
        "## Planner Action",
        "",
        f"- Name: `{action.get('name')}`",
        f"- Priority: `{action.get('priority')}`",
        f"- Command: `{action.get('command')}`",
        "",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_summary = resolve_source_summary(args.source_summary)
    payload = build_payload(source_summary)
    write_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
