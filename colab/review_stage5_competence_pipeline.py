"""Review the Stage 5 competence-preserving pipeline and print the next action.

This is intentionally CPU-only. It reads a competence-preserving pipeline
summary, extracts the important child-stage status, and delegates next-action
routing to ``plan_stage5_next_run`` so post-GPU triage stays consistent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from colab.plan_stage5_next_run import plan_next_actions
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    from plan_stage5_next_run import plan_next_actions


ROOT = Path(__file__).resolve().parents[1]
REVIEW_KIND = "stage5_competence_pipeline_review"


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


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def default_source_summary() -> Path:
    pointer = current_source_summary_file()
    if pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw:
            return resolve_path(raw)
    return ROOT / "outputs" / "stage5" / "stage5_competence_recovery_from_reentry_benchmark" / "summary.json"


def nested_status(payload: dict[str, Any], key: str) -> str:
    child = payload.get(key)
    return str(child.get("status") or "") if isinstance(child, dict) else ""


def nested_best_checkpoint(payload: dict[str, Any], key: str) -> str:
    child = payload.get(key)
    if not isinstance(child, dict):
        return ""
    for candidate in (
        child.get("best_checkpoint"),
        child.get("best_arm", {}).get("best_checkpoint") if isinstance(child.get("best_arm"), dict) else None,
        child.get("balanced_assessment", {}).get("best_checkpoint")
        if isinstance(child.get("balanced_assessment"), dict)
        else None,
    ):
        if isinstance(candidate, dict) and candidate.get("checkpoint"):
            return str(candidate["checkpoint"]).replace("\\", "/")
    return str(child.get("checkpoint") or child.get("selected_checkpoint") or "").replace("\\", "/")


def build_review(source_summary: Path) -> dict[str, Any]:
    payload = read_json(source_summary)
    actions = plan_next_actions(payload, source_summary=source_summary)
    first_action = actions[0] if actions else {}
    return {
        "kind": REVIEW_KIND,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_summary": path_for_cli(source_summary),
        "source_kind": payload.get("kind"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "failed_stage": payload.get("failed_stage"),
        "failure_diagnosis": payload.get("failure_diagnosis"),
        "arc_mix_run_id": payload.get("arc_mix_run_id"),
        "arc_mix_status": payload.get("arc_mix_status") or nested_status(payload, "arc_mix"),
        "arc_mix_summary": payload.get("arc_mix_summary"),
        "arc_mix_best_checkpoint": nested_best_checkpoint(payload, "arc_mix"),
        "full_assessment_run_id": payload.get("full_assessment_run_id"),
        "full_assessment_status": payload.get("full_assessment_status") or nested_status(payload, "full_assessment"),
        "full_assessment_summary": payload.get("full_assessment_summary"),
        "full_assessment_best_checkpoint": nested_best_checkpoint(payload, "full_assessment"),
        "next_action": {
            "name": first_action.get("name"),
            "priority": first_action.get("priority"),
            "reason": first_action.get("reason"),
            "command": first_action.get("command"),
        },
    }


def report_lines(payload: dict[str, Any]) -> list[str]:
    action = payload.get("next_action") if isinstance(payload.get("next_action"), dict) else {}
    lines = [
        "# Stage 5 Competence Pipeline Review",
        "",
        f"- Source summary: `{payload.get('source_summary')}`",
        f"- Run id: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Failed stage: `{payload.get('failed_stage') or ''}`",
        f"- Failure diagnosis: `{payload.get('failure_diagnosis') or ''}`",
        f"- ARC-mix status: `{payload.get('arc_mix_status') or ''}`",
        f"- ARC-mix summary: `{payload.get('arc_mix_summary') or ''}`",
        f"- ARC-mix checkpoint: `{payload.get('arc_mix_best_checkpoint') or ''}`",
        f"- Full assessment status: `{payload.get('full_assessment_status') or ''}`",
        f"- Full assessment summary: `{payload.get('full_assessment_summary') or ''}`",
        f"- Full assessment checkpoint: `{payload.get('full_assessment_best_checkpoint') or ''}`",
        "",
        "## Next Action",
        f"- Name: `{action.get('name') or ''}`",
        f"- Priority: `{action.get('priority') or ''}`",
        f"- Reason: {action.get('reason') or ''}",
        f"- Command: `{action.get('command') or ''}`",
        "",
    ]
    return lines


def write_review(payload: dict[str, Any], *, run_id: str | None = None) -> Path:
    rid = run_id or time.strftime("stage5_competence_pipeline_review_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text("\n".join(report_lines(payload)), encoding="utf-8")
    return run_dir / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", default="", help="Competence pipeline summary path. Defaults to current pointer.")
    parser.add_argument("--run-id", default="", help="Optional output run id.")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write review artifacts.")
    args = parser.parse_args(argv)

    source_summary = resolve_path(args.source_summary) if args.source_summary else default_source_summary()
    payload = build_review(source_summary)
    print("\n".join(report_lines(payload)))
    if not args.no_write:
        out = write_review(payload, run_id=args.run_id or None)
        print(f"wrote_review={path_for_cli(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
