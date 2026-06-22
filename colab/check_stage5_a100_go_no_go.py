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
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.plan_stage5_next_run import (
    path_for_cli,
    plan_next_actions,
    read_json,
    resolve_source_summary,
    source_kind,
)
from colab.run_stage5_recovered_phase1_arc_gate import (
    candidate_drive_checkpoints,
    path_for_cli as checkpoint_path_for_cli,
)


RUN_ID = os.environ.get("STAGE5_A100_GO_NO_GO_RUN_ID") or time.strftime(
    "stage5_a100_go_no_go_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID


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


def infer_stage5_run_id(path: str | Path) -> str | None:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part == "stage5" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def checkpoint_from_payload(payload: dict[str, Any]) -> str | None:
    """Find the checkpoint the guarded next action is expected to consume."""

    direct = payload.get("selected_checkpoint") or payload.get("checkpoint")
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


def checkpoint_availability(payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint = checkpoint_from_payload(payload)
    if not checkpoint:
        return {
            "checkpoint": None,
            "available": False,
            "exists": False,
            "drive_candidate_exists": False,
            "reason": "No selected checkpoint was found in the source summary.",
        }

    checkpoint_path = Path(checkpoint)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path
    run_id = infer_stage5_run_id(checkpoint_path)
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


def apply_checkpoint_guard(
    decision: dict[str, Any],
    *,
    source_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = checkpoint_availability(source_payload)
    if not decision.get("go"):
        return decision, checkpoint
    if checkpoint.get("available"):
        return decision, checkpoint
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

    if source_has_calibration_warning(source_payload):
        return {
            "go": False,
            "status": "calibration_warning_no_go",
            "spend_class": "none",
            "reason": "Source summary reports calibration warning; inspect locally before using A100.",
        }

    if script == "colab/run_stage5_balanced_arc_mix_gate.py":
        return {
            "go": True,
            "status": "go_bounded_proxy",
            "spend_class": "single_arc_mix_proxy",
            "reason": "Planner recommends exactly one bounded competence-recovery proxy.",
        }

    if script == "colab/run_stage5_recovery_full_assessment.py":
        if source_status in {"proxy_lift", "proxy_matches_base"} and bool(source_payload.get("passed", False)):
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
        return {
            "go": True,
            "status": "go_broader_benchmark",
            "spend_class": "bounded_benchmark_suite",
            "reason": "Planner found a nonnegative balanced checkpoint and recommends broader benchmarks.",
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
    decision = classify_action(action, source_payload=source_payload)
    decision, checkpoint = apply_checkpoint_guard(decision, source_payload=source_payload)
    return {
        "run_id": RUN_ID,
        "kind": "stage5_a100_go_no_go",
        "source_summary": path_for_cli(source_summary),
        "source_kind": source_kind(source_payload),
        "source_status": source_payload.get("status"),
        "decision": decision,
        "checkpoint_preflight": checkpoint,
        "recommended_action": action,
        "all_actions": actions,
    }


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    action = payload.get("recommended_action") or {}
    decision = payload["decision"]
    checkpoint = payload.get("checkpoint_preflight") or {}
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
