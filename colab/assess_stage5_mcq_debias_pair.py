"""Combine ARC-Easy and ARC-Challenge MCQ debias diagnostics.

This is a no-GPU assessor. The single-dataset debias diagnostic answers whether
an apparent direct-route MCQ regression survives option-content and cyclic
permutation scoring. This script makes that gate explicit across ARC-Easy and
ARC-Challenge before any direct-preservation training is allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_MCQ_DEBIAS_PAIR_RUN_ID") or time.strftime(
    "stage5_mcq_debias_pair_%Y%m%d_%H%M%S"
)
DEFAULT_ARC_EASY_SUMMARY = Path("outputs/stage5/stage5_mcq_debias_direct_20260622_194346/summary.json")
CURRENT_SOURCE_SUMMARY_FILE = ROOT / "config" / "stage5_current_source_summary.txt"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def latest_arc_challenge_summary() -> Path | None:
    candidates: list[Path] = []
    stage5 = ROOT / "outputs" / "stage5"
    if not stage5.exists():
        return None
    for path in stage5.glob("**/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("kind") == "stage5_mcq_debias_diagnostic"
            and str(payload.get("arc_config")) == "ARC-Challenge"
        ):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0] if candidates else None


def debias_payload(payload: dict[str, Any], *, expected_arc_config: str) -> dict[str, Any]:
    if payload.get("kind") != "stage5_mcq_debias_diagnostic":
        raise ValueError(f"Expected stage5_mcq_debias_diagnostic, got {payload.get('kind')!r}")
    if str(payload.get("arc_config")) != expected_arc_config:
        raise ValueError(f"Expected arc_config={expected_arc_config!r}, got {payload.get('arc_config')!r}")
    return payload


def diagnostic_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return {
        "summary": path_for_cli(path),
        "run_id": payload.get("run_id"),
        "arc_config": payload.get("arc_config"),
        "arc_limit": payload.get("arc_limit"),
        "status": payload.get("status"),
        "passed": bool(payload.get("passed", False)),
        "label_delta": int(decision.get("label_delta", 0) or 0),
        "content_delta": int(decision.get("content_delta", 0) or 0),
        "cyclic_delta": int(decision.get("cyclic_delta", 0) or 0),
        "best_debiased_delta": int(decision.get("best_debiased_delta", 0) or 0),
        "closure_vs_label": int(decision.get("closure_vs_label", 0) or 0),
        "next_step": decision.get("next_step"),
    }


def debiased_gap_persists(row: dict[str, Any], *, max_debiased_gap: int) -> bool:
    if row["status"] == "content_degradation_persists":
        return True
    return min(row["cyclic_delta"], row["best_debiased_delta"]) < -max_debiased_gap


def selection_bias_resolved(row: dict[str, Any], *, max_debiased_gap: int, min_closure: int) -> bool:
    if row["status"] == "selection_bias_likely":
        return True
    debiased_ok = min(row["cyclic_delta"], row["best_debiased_delta"]) >= -max_debiased_gap
    closes_label_gap = row["closure_vs_label"] >= min_closure
    return debiased_ok and closes_label_gap


def assess_pair(
    *,
    arc_easy_path: Path,
    arc_easy_payload: dict[str, Any],
    arc_challenge_path: Path,
    arc_challenge_payload: dict[str, Any],
    max_debiased_gap: int,
    min_closure: int,
) -> dict[str, Any]:
    easy = diagnostic_row(arc_easy_path, arc_easy_payload)
    challenge = diagnostic_row(arc_challenge_path, arc_challenge_payload)
    rows = [easy, challenge]

    blocking = [row for row in rows if debiased_gap_persists(row, max_debiased_gap=max_debiased_gap)]
    resolved = [
        row
        for row in rows
        if selection_bias_resolved(row, max_debiased_gap=max_debiased_gap, min_closure=min_closure)
    ]
    if blocking:
        status = "mcq_content_gap_persists"
        passed = False
        reason = (
            "At least one ARC split still shows a recurrent/base gap after option-content and cyclic scoring; "
            "run a bounded direct-preservation probe against that split before more curriculum training."
        )
        next_step = "Run the bounded max_loops=1 direct-preservation probe from the blocking debias summary."
        blocking_summary = blocking[0]["summary"]
    elif len(resolved) == len(rows):
        status = "mcq_selection_bias_confirmed"
        passed = True
        reason = (
            "ARC-Easy and ARC-Challenge both indicate the apparent MCQ regression is primarily an option-ID "
            "selection-bias artifact under bare-label scoring."
        )
        next_step = (
            "Standardize MCQ benchmark reporting on content/cyclic permutation scoring before interpreting "
            "recurrent-vs-base deltas or launching more preservation training."
        )
        blocking_summary = None
    else:
        status = "mcq_debias_mixed_or_inconclusive"
        passed = False
        reason = "The ARC-Easy and ARC-Challenge debias diagnostics do not agree cleanly."
        next_step = "Inspect the per-row debias records before choosing training or scoring changes."
        blocking_summary = None

    return {
        "run_id": RUN_ID,
        "kind": "stage5_mcq_debias_pair_assessment",
        "status": status,
        "passed": passed,
        "reason": reason,
        "next_step": next_step,
        "thresholds": {
            "max_debiased_gap": max_debiased_gap,
            "min_closure": min_closure,
        },
        "diagnostics": rows,
        "blocking_summary": blocking_summary,
        "source_summaries": {
            "arc_easy": path_for_cli(arc_easy_path),
            "arc_challenge": path_for_cli(arc_challenge_path),
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Stage 5 MCQ Debias Pair Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Diagnostics",
    ]
    for row in payload["diagnostics"]:
        lines.extend(
            [
                "",
                f"### {row['arc_config']}",
                f"- Summary: `{row['summary']}`",
                f"- Status: `{row['status']}`",
                f"- Label delta: `{row['label_delta']}`",
                f"- Content delta: `{row['content_delta']}`",
                f"- Cyclic delta: `{row['cyclic_delta']}`",
                f"- Best debiased delta: `{row['best_debiased_delta']}`",
                f"- Closure vs label: `{row['closure_vs_label']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc_easy_summary", default=os.environ.get("STAGE5_MCQ_DEBIAS_ARC_EASY_SUMMARY", str(DEFAULT_ARC_EASY_SUMMARY)))
    parser.add_argument("--arc_challenge_summary", default=os.environ.get("STAGE5_MCQ_DEBIAS_ARC_CHALLENGE_SUMMARY", ""))
    parser.add_argument("--output_dir", default=os.environ.get("STAGE5_MCQ_DEBIAS_PAIR_OUTPUT_DIR", ""))
    parser.add_argument("--max_debiased_gap", type=int, default=int(os.environ.get("STAGE5_MCQ_DEBIAS_PAIR_MAX_DEBIASED_GAP", "2")))
    parser.add_argument("--min_closure", type=int, default=int(os.environ.get("STAGE5_MCQ_DEBIAS_PAIR_MIN_CLOSURE", "3")))
    parser.add_argument("--update_current_source", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arc_easy_path = resolve_path(args.arc_easy_summary)
    challenge_path = resolve_path(args.arc_challenge_summary) if args.arc_challenge_summary else latest_arc_challenge_summary()
    if challenge_path is None:
        raise SystemExit("No ARC-Challenge MCQ debias summary found. Pass --arc_challenge_summary.")

    easy_payload = debias_payload(read_json(arc_easy_path), expected_arc_config="ARC-Easy")
    challenge_payload = debias_payload(read_json(challenge_path), expected_arc_config="ARC-Challenge")
    payload = assess_pair(
        arc_easy_path=arc_easy_path,
        arc_easy_payload=easy_payload,
        arc_challenge_path=challenge_path,
        arc_challenge_payload=challenge_payload,
        max_debiased_gap=args.max_debiased_gap,
        min_closure=args.min_closure,
    )

    out_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "stage5" / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    write_json(summary_json, payload)
    write_report(summary_md, payload)
    if args.update_current_source:
        CURRENT_SOURCE_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_SOURCE_SUMMARY_FILE.write_text(path_for_cli(summary_json) + "\n", encoding="utf-8")

    print(summary_md.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
