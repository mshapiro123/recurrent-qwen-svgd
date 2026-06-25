"""Review Stage 5 re-entry diagnostics and print the next allowed action.

This is intentionally CPU-only. The re-entry repair sequence has mandatory
readout pauses after Stage 1, Stage 2, and Stage 3, so this script turns the
latest committed artifacts into a deterministic go/no-go recommendation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEW_KIND = "stage5_reentry_review"
REENTRY_SUMMARY_KINDS = {
    "reentry_drift_diagnostic",
    "stage5_reentry_norm_eval_only",
    "stage5_reentry_repair_smoke",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assessment_paths(root: Path | None = None) -> list[Path]:
    scan = root or (ROOT / "outputs" / "stage5")
    if not scan.exists():
        return []
    return sorted(scan.rglob("reentry_assessment.json"))


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def current_pointer_assessment(pointer: Path | None = None) -> dict[str, Any]:
    if pointer is None:
        return {"expected": False}
    pointer_path = pointer
    if not pointer_path.exists():
        return {"expected": False}
    raw = pointer_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"expected": False}

    pointed = resolve_path(raw)
    if not pointed.exists():
        return {
            "expected": True,
            "path": path_for_cli(pointed),
            "error": "current_pointer_target_missing",
        }

    try:
        payload = read_json(pointed)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "expected": True,
            "path": path_for_cli(pointed),
            "error": f"current_pointer_unreadable:{type(exc).__name__}",
        }

    if pointed.name == "reentry_assessment.json":
        if payload.get("kind") != "stage5_reentry_assessment":
            return {"expected": False}
        return {
            "expected": True,
            "assessment_path": pointed,
            "source_kind": payload.get("source_kind"),
            "payload": payload,
            "error": "",
        }

    if pointed.name != "summary.json" or payload.get("kind") not in REENTRY_SUMMARY_KINDS:
        return {"expected": False}

    assessment_path = pointed.with_name("reentry_assessment.json")
    if not assessment_path.exists():
        return {
            "expected": True,
            "path": path_for_cli(assessment_path),
            "source_kind": payload.get("kind"),
            "error": "current_pointer_assessment_missing",
        }

    try:
        assessment = read_json(assessment_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "expected": True,
            "path": path_for_cli(assessment_path),
            "source_kind": payload.get("kind"),
            "error": f"current_pointer_assessment_unreadable:{type(exc).__name__}",
        }
    if assessment.get("kind") != "stage5_reentry_assessment":
        return {
            "expected": True,
            "path": path_for_cli(assessment_path),
            "source_kind": payload.get("kind"),
            "error": "current_pointer_assessment_wrong_kind",
        }
    return {
        "expected": True,
        "assessment_path": assessment_path,
        "source_kind": assessment.get("source_kind"),
        "payload": assessment,
        "error": "",
    }


def latest_by_source_kind(
    paths: list[Path],
    *,
    pointer: Path | None = None,
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[tuple[float, Path, dict[str, Any]]]] = {}
    for path in paths:
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("kind") != "stage5_reentry_assessment":
            continue
        source_kind = str(payload.get("source_kind") or "")
        if not source_kind:
            continue
        grouped.setdefault(source_kind, []).append((path.stat().st_mtime, path, payload))
    latest = {
        source_kind: (path, payload)
        for source_kind, values in grouped.items()
        for _mtime, path, payload in [max(values, key=lambda item: item[0])]
    }
    pointer_info = current_pointer_assessment(pointer)
    if pointer_info.get("payload") and pointer_info.get("assessment_path") and pointer_info.get("source_kind"):
        latest[str(pointer_info["source_kind"])] = (
            pointer_info["assessment_path"],
            pointer_info["payload"],
        )
        pointer_info["preferred"] = True
    else:
        pointer_info["preferred"] = False
    return latest, pointer_info


def classify(grouped: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    drift = grouped.get("reentry_drift_diagnostic")
    norm = grouped.get("stage5_reentry_norm_eval_only")
    repair = grouped.get("stage5_reentry_repair_smoke")

    if repair is not None:
        repair_path, repair_payload = repair
        recommendation = str(repair_payload.get("recommendation") or "")
        if recommendation == "run_bounded_recovery_training_with_reentry_repair":
            action = "run_bounded_recovery_training_with_reentry_repair"
            target = "reentry_recovery_training"
            next_step = "Stage 3 passed; implement or launch the bounded recovery-training target."
        elif recommendation == "fix_loop1_preservation_eval_before_recovery_training":
            action = "stop_loop1_preservation_evidence_missing"
            target = ""
            next_step = "Stage 3 did not produce comparable loop-1 preservation evidence; fix the preservation eval before recovery training."
        elif recommendation == "review_or_reduce_repair_lr_before_recovery_training":
            action = "stop_loop1_regression"
            target = ""
            next_step = "Stage 3 harmed loop-1 preservation; reduce repair LR or change repair target before training."
        elif recommendation == "fix_reentry_adapter_before_recovery_training":
            action = "stop_reentry_adapter_not_live"
            target = ""
            next_step = "Stage 3 did not produce live re-entry adapter gradients; fix adapter wiring before training."
        elif recommendation == "extend_reentry_repair_smoke_or_increase_adapter_lr":
            action = "extend_reentry_adapter_smoke"
            target = "reentry_repair_smoke"
            next_step = "Re-entry adapter gradients are live but movement was too small; rerun a bounded Stage 3 variant."
        elif recommendation == "extend_reentry_repair_smoke_or_increase_bridge_lr":
            if str(repair_payload.get("status") or "") == "bridge_gate_collapsed":
                action = "extend_repair_smoke_bridge_gate_active"
                next_step = (
                    "Bridge projection moved or stayed live, but bridge_gate collapsed; "
                    "rerun a bounded Stage 3 variant and inspect whether the gate should "
                    "be held active during the smoke."
                )
            else:
                action = "extend_repair_smoke"
                next_step = "Bridge gradients are live but movement was too small; rerun a bounded Stage 3 variant."
            target = "reentry_repair_smoke"
        else:
            action = "stop_repair_failed"
            target = ""
            next_step = "Stage 3 did not repair bridge liveness; inspect repair controls before more GPU."
        return {
            "action": action,
            "next_target": target,
            "next_step": next_step,
            "latest_stage": "stage3_repair_smoke",
            "latest_assessment": path_for_cli(repair_path),
            "latest_status": repair_payload.get("status"),
            "latest_recommendation": recommendation,
        }

    if norm is not None:
        norm_path, norm_payload = norm
        recommendation = str(norm_payload.get("recommendation") or "")
        if recommendation == "run_reentry_repair_smoke":
            action = "run_reentry_repair_smoke"
            target = "reentry_repair_smoke"
            next_step = "Stage 2 did not show a major eval-only regression; run Stage 3 repair smoke."
        else:
            action = "stop_norm_regression"
            target = ""
            next_step = "Stage 2 did not clear repair smoke; review norm/candidate regression before training."
        return {
            "action": action,
            "next_target": target,
            "next_step": next_step,
            "latest_stage": "stage2_norm",
            "latest_assessment": path_for_cli(norm_path),
            "latest_status": norm_payload.get("status"),
            "latest_recommendation": recommendation,
        }

    if drift is not None:
        drift_path, drift_payload = drift
        recommendation = str(drift_payload.get("recommendation") or "")
        target = "reentry_norm_diagnostic" if "norm" in recommendation else "reentry_repair_smoke"
        return {
            "action": recommendation or "run_reentry_norm_diagnostic",
            "next_target": target,
            "next_step": "Stage 1 is complete; run Stage 2 eval-only re-entry normalization before trainable repair.",
            "latest_stage": "stage1_drift",
            "latest_assessment": path_for_cli(drift_path),
            "latest_status": drift_payload.get("status"),
            "latest_recommendation": recommendation,
        }

    return {
        "action": "run_reentry_drift_diagnostic",
        "next_target": "reentry_drift_diagnostic",
        "next_step": "No re-entry assessment artifacts found; run Stage 1 drift diagnostic.",
        "latest_stage": "none",
        "latest_assessment": None,
        "latest_status": None,
        "latest_recommendation": None,
    }


def launch_env_for_decision(decision: dict[str, Any]) -> dict[str, str]:
    """Return pasteable Colab env for the next allowed re-entry action.

    Stop decisions intentionally return an empty dict. Bounded Stage 3 retry
    decisions include small, explicit knob changes so a retry is not an
    ad hoc rerun of the same failed smoke.
    """

    target = str(decision.get("next_target") or "")
    action = str(decision.get("action") or "")
    if not target:
        return {}
    env = {"STAGE5_CURRENT_A100_TARGET": target}
    if action == "extend_reentry_adapter_smoke":
        env.update(
            {
                "STAGE5_REENTRY_REPAIR_MAX_STEPS": "50",
                "STAGE5_REENTRY_REPAIR_LR": "2e-5",
                "STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES": "bridge,reentry,halt",
            }
        )
    elif action in {"extend_repair_smoke", "extend_repair_smoke_bridge_gate_active"}:
        env.update(
            {
                "STAGE5_REENTRY_REPAIR_MAX_STEPS": "50",
                "STAGE5_REENTRY_REPAIR_LR": "2e-5",
                "STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES": "bridge,reentry,halt",
            }
        )
    return env


def pointer_error_review(pointer_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": REVIEW_KIND,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assessments": {},
        "current_pointer": {
            "expected": pointer_info.get("expected"),
            "path": pointer_info.get("path"),
            "source_kind": pointer_info.get("source_kind"),
            "error": pointer_info.get("error"),
            "preferred": False,
        },
        "action": "fix_current_pointer_reentry_assessment",
        "next_target": "",
        "next_step": "Current source pointer names a re-entry artifact, but its assessment is missing or unreadable; recover or rerun that stage before spending GPU.",
        "latest_stage": "current_pointer_error",
        "latest_assessment": pointer_info.get("path"),
        "latest_status": None,
        "latest_recommendation": pointer_info.get("error"),
        "launch_env": {},
    }


def build_review(paths: list[Path], *, pointer: Path | None = None) -> dict[str, Any]:
    grouped, pointer_info = latest_by_source_kind(paths, pointer=pointer)
    if pointer_info.get("expected") and pointer_info.get("error"):
        return pointer_error_review(pointer_info)
    decision = classify(grouped)
    launch_env = launch_env_for_decision(decision)
    assessments = {
        kind: {
            "path": path_for_cli(path),
            "status": payload.get("status"),
            "recommendation": payload.get("recommendation"),
            "source_run_id": payload.get("source_run_id"),
        }
        for kind, (path, payload) in grouped.items()
    }
    return {
        "kind": REVIEW_KIND,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assessments": assessments,
        "current_pointer": {
            "expected": pointer_info.get("expected"),
            "path": path_for_cli(pointer_info["assessment_path"]) if pointer_info.get("assessment_path") else pointer_info.get("path"),
            "source_kind": pointer_info.get("source_kind"),
            "error": pointer_info.get("error"),
            "preferred": pointer_info.get("preferred"),
        },
        **decision,
        "launch_env": launch_env,
    }


def report_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Stage 5 Re-entry Review",
        "",
        f"- Latest stage: `{payload.get('latest_stage')}`",
        f"- Latest assessment: `{payload.get('latest_assessment') or 'none'}`",
        f"- Latest status: `{payload.get('latest_status')}`",
        f"- Latest recommendation: `{payload.get('latest_recommendation')}`",
        f"- Action: `{payload.get('action')}`",
        f"- Next target: `{payload.get('next_target') or 'none'}`",
        f"- Next step: {payload.get('next_step')}",
        "",
        "## Assessments",
    ]
    assessments = payload.get("assessments") if isinstance(payload.get("assessments"), dict) else {}
    if not assessments:
        lines.append("- none")
    else:
        for kind, row in sorted(assessments.items()):
            lines.append(
                f"- `{kind}`: status=`{row.get('status')}`, "
                f"recommendation=`{row.get('recommendation')}`, path=`{row.get('path')}`"
            )
    pointer = payload.get("current_pointer") if isinstance(payload.get("current_pointer"), dict) else {}
    if pointer:
        lines.extend(
            [
                "",
                "## Current Pointer",
                f"- Expected re-entry pointer: `{pointer.get('expected')}`",
                f"- Preferred pointer assessment: `{pointer.get('preferred')}`",
                f"- Source kind: `{pointer.get('source_kind')}`",
                f"- Path: `{pointer.get('path')}`",
                f"- Error: `{pointer.get('error') or ''}`",
            ]
        )
    launch_env = payload.get("launch_env") if isinstance(payload.get("launch_env"), dict) else {}
    if launch_env:
        lines.extend(["", "## Launch Env"])
        for key, value in sorted(launch_env.items()):
            lines.append(f"- `{key}={value}`")
    lines.append("")
    return lines


def write_review(payload: dict[str, Any], *, run_id: str | None = None) -> Path:
    run_name = run_id or f"stage5_reentry_review_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "outputs" / "stage5" / run_name
    summary_path = run_dir / "summary.json"
    write_json(summary_path, payload)
    (run_dir / "summary.md").write_text("\n".join(report_lines(payload)), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan_root", default="outputs/stage5")
    parser.add_argument("--output_run_id", default="")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()

    payload = build_review(assessment_paths(resolve_path(args.scan_root)), pointer=current_source_summary_file())
    print("\n".join(report_lines(payload)), flush=True)
    if not args.no_write:
        path = write_review(payload, run_id=args.output_run_id or None)
        print(f"review_summary={path_for_cli(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
