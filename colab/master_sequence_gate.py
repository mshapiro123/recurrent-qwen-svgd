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


def checkpoint_from_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        checkpoint = value.get("checkpoint")
        if checkpoint:
            return str(checkpoint)
    return None


def checkpoint_value_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract a recurrent adapter checkpoint from known Stage 5 payload shapes."""

    candidates = [
        (payload.get("metadata") or {}).get("recovered_checkpoint")
        if isinstance(payload.get("metadata"), dict)
        else None,
        (payload.get("metadata") or {}).get("checkpoint")
        if isinstance(payload.get("metadata"), dict)
        else None,
        (payload.get("compact") or {}).get("final_checkpoint")
        if isinstance(payload.get("compact"), dict)
        else None,
        (payload.get("autopilot_compact") or {}).get("final_checkpoint")
        if isinstance(payload.get("autopilot_compact"), dict)
        else None,
        payload.get("final_checkpoint"),
        payload.get("tuned_checkpoint"),
        payload.get("checkpoint"),
        payload.get("phase1_checkpoint"),
        checkpoint_from_value(payload.get("selected_checkpoint")),
        checkpoint_from_value(payload.get("best_checkpoint")),
        checkpoint_from_value((payload.get("best_arm") or {}).get("best_checkpoint"))
        if isinstance(payload.get("best_arm"), dict)
        else None,
    ]
    stages = payload.get("stages")
    if isinstance(stages, list) and stages:
        last_stage = stages[-1]
        if isinstance(last_stage, dict):
            candidates.append(checkpoint_from_value(last_stage.get("selected_checkpoint")))
    curriculum = payload.get("curriculum")
    if isinstance(curriculum, dict):
        candidates.append(curriculum.get("final_checkpoint"))
        curriculum_stages = curriculum.get("stages")
        if isinstance(curriculum_stages, list) and curriculum_stages:
            last_stage = curriculum_stages[-1]
            if isinstance(last_stage, dict):
                candidates.append(checkpoint_from_value(last_stage.get("selected_checkpoint")))
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def summary_links_from_payload(payload: dict[str, Any]) -> list[str]:
    """Return likely upstream summary links in current-to-source order."""

    links: list[str] = []
    for key in (
        "recurrent_summary",
        "recurrent_benchmark_summary",
        "benchmark_summary",
        "benchmark_source_summary",
        "benchmark_assessment_summary",
        "nested_source_summary",
        "source_summary",
        "curriculum_source_summary",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            links.append(value)

    assessment = payload.get("recipe_control_assessment")
    if isinstance(assessment, dict):
        for key in ("recurrent_summary", "recurrent_benchmark_summary", "benchmark_summary", "source_summary"):
            value = assessment.get(key)
            if isinstance(value, str) and value.strip():
                links.append(value)

    source_summaries = payload.get("source_summaries")
    if isinstance(source_summaries, dict):
        for key in ("arc_challenge", "arc_easy"):
            value = source_summaries.get(key)
            if isinstance(value, str) and value.strip():
                links.append(value)
    return links


def phase1_checkpoint_for_breadth(
    payload: dict[str, Any],
    *,
    root: Path,
    summary_path: Path | None = None,
    depth: int = 0,
    seen: set[Path] | None = None,
) -> dict[str, Any] | None:
    """Resolve the recurrent checkpoint Phase 2 should diagnose.

    The current pointer may be a same-recipe dense-control assessment rather
    than the recurrent benchmark suite or Stage 4 recovery summary. This walks
    those evidence links back to the checkpoint-bearing recurrent artifact.
    """

    checkpoint = checkpoint_value_from_payload(payload)
    if checkpoint:
        return {
            "checkpoint": checkpoint,
            "checkpoint_source_summary": str(summary_path) if summary_path else None,
            "checkpoint_source_kind": payload.get("kind"),
            "checkpoint_source_gate": payload.get("gate"),
        }
    if summary_path is not None:
        adjacent = summary_path.parent / "recurrent_adapter_checkpoint.pt"
        if adjacent.exists():
            return {
                "checkpoint": str(adjacent),
                "checkpoint_source_summary": str(summary_path),
                "checkpoint_source_kind": payload.get("kind"),
                "checkpoint_source_gate": payload.get("gate"),
            }

    if depth >= 8:
        return None
    seen = seen or set()
    if summary_path is not None:
        resolved = summary_path.resolve()
        if resolved in seen:
            return None
        seen.add(resolved)

    for link in summary_links_from_payload(payload):
        path = resolve_path(root, link)
        if not path.exists():
            continue
        try:
            linked_payload = read_json(path)
        except Exception:
            continue
        resolved = phase1_checkpoint_for_breadth(
            linked_payload,
            root=root,
            summary_path=path,
            depth=depth + 1,
            seen=seen,
        )
        if resolved:
            return resolved
    return None


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
        checkpoint_info = phase1_checkpoint_for_breadth(payload, root=root, summary_path=summary_path)
        if not checkpoint_info:
            raise SystemExit(
                f"{action_name} passed the Phase 1 gate but could not resolve the recurrent "
                f"checkpoint to diagnose from {summary_path}. Run from a source summary that "
                "links back to the Stage 4 recurrent checkpoint, or set an explicit checkpoint "
                "environment override for an intentional archaeology run."
            )
        return {
            "allowed": True,
            "source_summary": str(summary_path),
            "source_kind": payload.get("kind"),
            "source_gate": payload.get("gate"),
            "source_status": payload.get("status"),
            **checkpoint_info,
        }

    raise SystemExit(
        f"{action_name} is blocked by the master sequence. "
        "Run Stage 3 `reentry_repair_smoke`, Stage 4 `reentry_recovery_training`, "
        "then `debiased_benchmark_suite` and `dense_mcq_trace_sft_control` first. "
        f"Current source summary is {summary_path} with kind={payload.get('kind')!r}, "
        f"gate={payload.get('gate')!r}, status={payload.get('status')!r}. "
        f"Set {allow_env}=1 only for an intentional pre-Phase-1 archaeology run."
    )
