"""Activate the Stage 5 debiased MCQ scoring policy.

This is a no-GPU bridge after ARC-Easy and ARC-Challenge both confirm that the
apparent recurrent/base MCQ gap is mostly a bare option-label artifact. It does
not change model weights. It records the scoring policy that future MCQ claims
must use and flags stale label-only artifacts that should not drive training.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_MCQ_SCORING_POLICY_RUN_ID") or time.strftime(
    "stage5_mcq_scoring_policy_%Y%m%d_%H%M%S"
)
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


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def current_source_summary() -> Path:
    value = CURRENT_SOURCE_SUMMARY_FILE.read_text(encoding="utf-8").strip()
    if not value:
        raise FileNotFoundError("config/stage5_current_source_summary.txt is empty.")
    return resolve_path(value)


def stale_mcq_artifacts() -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    root = ROOT / "outputs" / "stage5"
    if not root.exists():
        return stale
    for path in root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if not payload:
            continue
        kind = str(payload.get("kind") or "")
        score_target = str(payload.get("score_target") or "")
        if kind == "stage5_balanced_mcq_checkpoint_assessment" and score_target == "label":
            stale.append(
                {
                    "summary": path_for_cli(path),
                    "kind": kind,
                    "reason": "Balanced MCQ checkpoint assessment used bare-label score_target=label.",
                }
            )
        elif kind == "stage5_benchmark_suite":
            benchmarks = {str(item) for item in payload.get("benchmarks", [])}
            results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
            mcq_benchmarks = sorted(item for item in benchmarks if item in {"arc_easy", "arc_challenge", "gpqa_lite", "gpqa"})
            label_only = [
                bench
                for bench in mcq_benchmarks
                if isinstance(results.get(bench), dict)
                and "label" in results[bench]
                and not ({"cyclic_label_aggregated", "content_question_only"} & set(results[bench]))
            ]
            if label_only:
                stale.append(
                    {
                        "summary": path_for_cli(path),
                        "kind": kind,
                        "benchmarks": label_only,
                        "reason": "Benchmark suite has MCQ label results without debiased content/cyclic companions.",
                    }
                )
    return stale


def build_policy(*, source_summary: Path, source_payload: dict[str, Any]) -> dict[str, Any]:
    status = str(source_payload.get("status") or "unknown")
    if source_payload.get("kind") != "stage5_mcq_debias_pair_assessment":
        raise ValueError(f"Expected stage5_mcq_debias_pair_assessment, got {source_payload.get('kind')!r}")
    if status != "mcq_selection_bias_confirmed":
        return {
            "run_id": RUN_ID,
            "kind": "stage5_mcq_scoring_policy",
            "status": "policy_not_activated",
            "passed": False,
            "source_summary": path_for_cli(source_summary),
            "source_status": status,
            "reason": "Debiased MCQ scoring policy activates only after paired ARC-Easy/ARC-Challenge confirmation.",
            "next_step": "Use the paired debias assessment's action instead of standardizing MCQ claims.",
        }

    stale = stale_mcq_artifacts()
    return {
        "run_id": RUN_ID,
        "kind": "stage5_mcq_scoring_policy",
        "status": "debiased_mcq_policy_active",
        "passed": True,
        "source_summary": path_for_cli(source_summary),
        "source_status": status,
        "primary_metrics": ["cyclic_label_aggregated", "content_question_only"],
        "diagnostic_only_metrics": ["label"],
        "requirements": [
            "Do not use bare A/B/C/D accuracy alone for recurrent-vs-base MCQ claims.",
            "Report cyclic-permutation aggregated score as the primary MCQ option-likelihood metric.",
            "Report content-question-only score and label prediction drift as diagnostics.",
            "Do not launch direct-preservation training from a bare-label MCQ regression.",
            "Apply any later conditional-invariance objective only to depth>=2 reasoning-path training.",
        ],
        "stale_label_only_artifacts": stale,
        "reason": (
            "Paired ARC-Easy and ARC-Challenge debias diagnostics indicate that bare-label MCQ deltas are "
            "not a reliable training target for this recurrent model."
        ),
        "next_step": (
            "Regenerate or annotate any stale MCQ benchmark claims under cyclic/content scoring before using "
            "them for training, release, GPQA, or scale-up decisions."
        ),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Stage 5 MCQ Scoring Policy - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source summary: `{payload.get('source_summary')}`",
        f"- Reason: {payload.get('reason')}",
        f"- Next step: {payload.get('next_step')}",
    ]
    if payload.get("primary_metrics"):
        lines.extend(
            [
                "",
                "## Policy",
                "",
                f"- Primary metrics: `{payload['primary_metrics']}`",
                f"- Diagnostic-only metrics: `{payload['diagnostic_only_metrics']}`",
                "",
                "## Requirements",
            ]
        )
        lines.extend(f"- {item}" for item in payload.get("requirements", []))
    stale = payload.get("stale_label_only_artifacts") or []
    lines.extend(["", "## Stale Label-Only Artifacts"])
    if stale:
        for row in stale:
            lines.append(f"- `{row.get('summary')}`: {row.get('reason')}")
    else:
        lines.append("- None found in local `outputs/stage5` scan.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_summary", default=os.environ.get("STAGE5_MCQ_SCORING_POLICY_SOURCE_SUMMARY", ""))
    parser.add_argument("--output_dir", default=os.environ.get("STAGE5_MCQ_SCORING_POLICY_OUTPUT_DIR", ""))
    parser.add_argument("--update_current_source", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_summary = resolve_path(args.source_summary) if args.source_summary else current_source_summary()
    payload = build_policy(source_summary=source_summary, source_payload=read_json(source_summary))
    output_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "stage5" / RUN_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summary_md, payload)
    if args.update_current_source:
        CURRENT_SOURCE_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_SOURCE_SUMMARY_FILE.write_text(path_for_cli(summary_json) + "\n", encoding="utf-8")
    print(summary_md.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
