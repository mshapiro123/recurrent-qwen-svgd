"""Small guards that keep Stage 5 Colab actions on the master sequence.

These guards are intentionally conservative. Breadth and particle diagnostics
are useful only after deterministic depth recovery has produced evidence that
the recurrent architecture beats the same data recipe's dense control without
losing the broader base-vs-recurrent benchmark gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y"}


def env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def current_source_summary(root: Path) -> Path:
    pointer = root / "config" / "stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Empty current source pointer: {pointer}")
    return resolve_path(root, value)


def broader_benchmark_gate_passed(payload: dict[str, Any]) -> bool:
    return (
        payload.get("gate") == "stage5_broader_benchmark_suite"
        and payload.get("status") == "passed"
        and payload.get("passed") is True
    )


def same_recipe_architecture_gate_passed(payload: dict[str, Any]) -> bool:
    return (
        payload.get("gate") == "stage5_same_recipe_mcq_architecture"
        and payload.get("status") == "hard_tail_lift_vs_dense"
        and payload.get("passed") is True
    )


def embedded_same_recipe_gate_passed(payload: dict[str, Any]) -> bool:
    assessment = payload.get("recipe_control_assessment")
    return (
        payload.get("kind") == "stage5_dense_mcq_trace_sft_control"
        and isinstance(assessment, dict)
        and assessment.get("ran") is True
        and assessment.get("status") == "hard_tail_lift_vs_dense"
        and assessment.get("passed") is True
    )


def phase1_depth_gate_passed(payload: dict[str, Any], *, root: Path, depth: int = 0) -> bool:
    """Return true only for evidence strong enough to unlock Phase 2.

    Accepted evidence is either a direct same-recipe architecture assessment or
    a dense-control summary embedding that assessment. For dense summaries we
    also require the upstream broader base-vs-recurrent benchmark gate to have
    passed, because hard-tail lift is not useful if easy/base preservation is
    still broken.
    """

    if same_recipe_architecture_gate_passed(payload):
        return True
    if not embedded_same_recipe_gate_passed(payload):
        return False

    if depth >= 5:
        return False
    candidates = [
        payload.get("source_summary"),
        payload.get("benchmark_assessment_summary"),
    ]
    for item in candidates:
        if not item:
            continue
        try:
            upstream = read_json(resolve_path(root, item))
        except Exception:
            continue
        if broader_benchmark_gate_passed(upstream):
            return True
        if phase1_depth_gate_passed(upstream, root=root, depth=depth + 1):
            return True
    return False


def require_phase1_depth_gate_for_breadth(
    *,
    root: Path,
    action_name: str,
    allow_env: str = "STAGE5_ALLOW_PRE_PHASE1_BREADTH",
) -> dict[str, Any]:
    """Block Phase 2/3 diagnostic actions until Phase 1 evidence exists."""

    if env_true(allow_env):
        return {"allowed": True, "override": allow_env}

    summary_path = current_source_summary(root)
    payload = read_json(summary_path)
    if phase1_depth_gate_passed(payload, root=root):
        return {
            "allowed": True,
            "source_summary": str(summary_path),
            "source_kind": payload.get("kind"),
            "source_gate": payload.get("gate"),
            "source_status": payload.get("status"),
        }

    raise SystemExit(
        f"{action_name} is blocked by the master sequence. "
        "Run Stage 3 `reentry_repair_smoke`, Stage 4 `reentry_recovery_training`, "
        "then `debiased_benchmark_suite` and `dense_mcq_trace_sft_control` first. "
        f"Current source summary is {summary_path} with kind={payload.get('kind')!r}, "
        f"gate={payload.get('gate')!r}, status={payload.get('status')!r}. "
        f"Set {allow_env}=1 only for an intentional pre-Phase-1 archaeology run."
    )
