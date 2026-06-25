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


def latest_by_source_kind(paths: list[Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
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
    return {
        source_kind: (path, payload)
        for source_kind, values in grouped.items()
        for _mtime, path, payload in [max(values, key=lambda item: item[0])]
    }


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
            action = "extend_repair_smoke"
            target = "reentry_repair_smoke"
            next_step = "Bridge gradients are live but movement was too small; rerun a bounded Stage 3 variant."
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


def build_review(paths: list[Path]) -> dict[str, Any]:
    grouped = latest_by_source_kind(paths)
    decision = classify(grouped)
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
        **decision,
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

    payload = build_review(assessment_paths(resolve_path(args.scan_root)))
    print("\n".join(report_lines(payload)), flush=True)
    if not args.no_write:
        path = write_review(payload, run_id=args.output_run_id or None)
        print(f"review_summary={path_for_cli(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
