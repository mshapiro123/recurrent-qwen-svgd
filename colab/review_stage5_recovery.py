"""Review Stage 5 re-entry recovery training before benchmark/control spend.

This CPU-only readout is the Stage 4 companion to ``review_stage5_reentry``.
It answers one narrow question: did the deterministic recovery run publish a
usable checkpoint with sane validation, so the next allowed GPU action is the
debiased benchmark suite?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEW_KIND = "stage5_reentry_recovery_review"
RECOVERY_KIND = "stage5_reentry_recovery_training"


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


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def summary_candidates(scan_root: Path | None = None, *, pointer: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if pointer and pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw:
            candidates.append(resolve_path(raw))
    scan = scan_root or (ROOT / "outputs" / "stage5")
    if scan.exists():
        candidates.extend(sorted(scan.rglob("summary.json"), key=lambda path: path.stat().st_mtime, reverse=True))
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = candidate.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def latest_recovery_summary(scan_root: Path | None = None, *, pointer: Path | None = None) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
    pointer_info: dict[str, Any] = {"expected": False, "preferred": False}
    for index, candidate in enumerate(summary_candidates(scan_root, pointer=pointer)):
        if not candidate.exists():
            if index == 0 and pointer and pointer.exists():
                pointer_info = {
                    "expected": True,
                    "preferred": False,
                    "path": path_for_cli(candidate),
                    "error": "current_pointer_target_missing",
                }
            continue
        try:
            payload = read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if index == 0 and pointer and pointer.exists():
                pointer_info = {
                    "expected": True,
                    "preferred": False,
                    "path": path_for_cli(candidate),
                    "error": f"current_pointer_unreadable:{type(exc).__name__}",
                }
            continue
        if index == 0 and pointer and pointer.exists():
            pointer_info = {
                "expected": True,
                "preferred": payload.get("kind") == RECOVERY_KIND,
                "path": path_for_cli(candidate),
                "kind": payload.get("kind"),
                "error": "" if payload.get("kind") == RECOVERY_KIND else "current_pointer_not_recovery_summary",
            }
        if payload.get("kind") == RECOVERY_KIND:
            return candidate, payload, pointer_info
    return None, None, pointer_info


def launch_env_for_summary(path: Path) -> dict[str, str]:
    return {
        "STAGE5_CURRENT_A100_TARGET": "debiased_benchmark_suite",
        "STAGE5_CURRENT_A100_SOURCE_SUMMARY": path_for_cli(path),
    }


def build_review(scan_root: Path | None = None, *, pointer: Path | None = None) -> dict[str, Any]:
    summary_path, payload, pointer_info = latest_recovery_summary(scan_root, pointer=pointer)
    if payload is None or summary_path is None:
        return {
            "kind": REVIEW_KIND,
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latest_stage": "none",
            "latest_summary": None,
            "latest_status": None,
            "action": "wait_for_reentry_recovery_training",
            "next_target": "",
            "next_step": "No Stage 4 recovery summary found; complete reentry_recovery_training after Stage 3 passes.",
            "current_pointer": pointer_info,
            "launch_env": {},
        }

    checkpoint = str(payload.get("phase1_checkpoint") or payload.get("checkpoint") or "").strip()
    validation_checks = payload.get("validation_checks") if isinstance(payload.get("validation_checks"), dict) else {}
    validation_status = str(validation_checks.get("status") or payload.get("status") or "")
    issues = validation_checks.get("issues") if isinstance(validation_checks.get("issues"), list) else []

    if not checkpoint:
        action = "stop_recovery_checkpoint_missing"
        target = ""
        next_step = "Stage 4 recovery summary is missing a checkpoint path; inspect wrapper and child SFT summaries before benchmarking."
        launch_env: dict[str, str] = {}
    elif validation_status and validation_status != "validation_sane":
        action = "stop_recovery_validation_needs_review"
        target = ""
        next_step = f"Stage 4 validation status is {validation_status!r} with issues {issues}; do not benchmark or run dense control yet."
        launch_env = {}
    else:
        action = "run_debiased_benchmark_suite"
        target = "debiased_benchmark_suite"
        next_step = "Stage 4 deterministic recovery is benchmark-ready; run debiased benchmark before dense control or breadth/SVGD."
        launch_env = launch_env_for_summary(summary_path)

    return {
        "kind": REVIEW_KIND,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest_stage": "stage4_reentry_recovery",
        "latest_summary": path_for_cli(summary_path),
        "latest_status": validation_status or payload.get("status"),
        "checkpoint": checkpoint,
        "validation_issues": issues,
        "action": action,
        "next_target": target,
        "next_step": next_step,
        "current_pointer": pointer_info,
        "launch_env": launch_env,
    }


def report_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Stage 5 Re-entry Recovery Review",
        "",
        f"- Latest stage: `{payload.get('latest_stage')}`",
        f"- Latest summary: `{payload.get('latest_summary') or 'none'}`",
        f"- Latest status: `{payload.get('latest_status')}`",
        f"- Checkpoint: `{payload.get('checkpoint') or ''}`",
        f"- Validation issues: `{payload.get('validation_issues') or []}`",
        f"- Action: `{payload.get('action')}`",
        f"- Next target: `{payload.get('next_target') or 'none'}`",
        f"- Next step: {payload.get('next_step')}",
    ]
    pointer = payload.get("current_pointer") if isinstance(payload.get("current_pointer"), dict) else {}
    if pointer:
        lines.extend(
            [
                "",
                "## Current Pointer",
                f"- Expected pointer: `{pointer.get('expected')}`",
                f"- Preferred pointer: `{pointer.get('preferred')}`",
                f"- Kind: `{pointer.get('kind')}`",
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
    run_name = run_id or f"stage5_reentry_recovery_review_{time.strftime('%Y%m%d_%H%M%S')}"
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

    payload = build_review(resolve_path(args.scan_root), pointer=current_source_summary_file())
    print("\n".join(report_lines(payload)), flush=True)
    if not args.no_write:
        path = write_review(payload, run_id=args.output_run_id or None)
        print(f"review_summary={path_for_cli(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
