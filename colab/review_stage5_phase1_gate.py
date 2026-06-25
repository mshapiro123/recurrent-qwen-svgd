"""Review the Phase 1 deterministic-depth architecture gate.

This CPU-only reviewer synthesizes the two artifacts needed before returning
to breadth/particles:

1. broader base-vs-recurrent benchmark assessment;
2. dense same-curriculum MCQ control assessment.

It deliberately separates "recurrent recovered base performance" from
"recurrent beat a dense same-recipe control", because the latter is the
architecture contribution test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEW_KIND = "stage5_phase1_gate_review"
BENCHMARK_GATE = "stage5_broader_benchmark_suite"
DENSE_GATE = "stage5_same_recipe_mcq_architecture"


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


def summary_paths(scan_root: Path | None = None) -> list[Path]:
    scan = scan_root or (ROOT / "outputs" / "stage5")
    if not scan.exists():
        return []
    return sorted(scan.rglob("summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def latest_payload(
    *,
    scan_root: Path | None = None,
    gate: str | None = None,
    kind: str | None = None,
    source_summary: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    source_cli = path_for_cli(resolve_path(source_summary)) if source_summary else None
    for path in summary_paths(scan_root):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if gate and payload.get("gate") != gate:
            continue
        if kind and payload.get("kind") != kind:
            continue
        if source_cli:
            payload_source = payload.get("source_summary")
            if not isinstance(payload_source, str) or path_for_cli(resolve_path(payload_source)) != source_cli:
                continue
        return path, payload
    return None, None


def current_pointer_payload(pointer: Path | None = None) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
    pointer = pointer or current_source_summary_file()
    if not pointer.exists():
        return None, None, {"expected": False}
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None, None, {"expected": False}
    path = resolve_path(raw)
    if not path.exists():
        return None, None, {"expected": True, "path": path_for_cli(path), "error": "current_pointer_target_missing"}
    try:
        payload = read_json(path)
    except Exception as exc:
        return None, None, {
            "expected": True,
            "path": path_for_cli(path),
            "error": f"current_pointer_unreadable:{type(exc).__name__}",
        }
    return path, payload, {
        "expected": True,
        "path": path_for_cli(path),
        "kind": payload.get("kind"),
        "gate": payload.get("gate"),
        "error": "",
    }


def benchmark_passed(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("gate") == BENCHMARK_GATE and payload.get("passed") is True)


def benchmark_hard_delta(payload: dict[str, Any] | None) -> int | None:
    if not payload:
        return None
    for row in payload.get("benchmarks") or []:
        if isinstance(row, dict) and row.get("benchmark") == "arc_challenge":
            value = row.get("correct_delta_recurrent_vs_base")
            return int(value) if isinstance(value, (int, float)) else None
    return None


def launch_env_for_dense_control(benchmark_assessment_path: Path) -> dict[str, str]:
    return {
        "STAGE5_CURRENT_A100_TARGET": "dense_mcq_trace_sft_control",
        "STAGE5_CURRENT_A100_SOURCE_SUMMARY": path_for_cli(benchmark_assessment_path),
    }


def reentry_recovery_health_block_reason(payload: dict[str, Any]) -> str | None:
    checks = payload.get("post_reentry_health_checks")
    if not isinstance(checks, dict):
        return "Stage 4 recovery summary is missing post-recovery re-entry health checks."
    status = str(checks.get("status") or "")
    if status != "reentry_health_sane":
        return f"Stage 4 post-recovery re-entry health is `{status}` with issues `{checks.get('issues', [])}`."
    return None


def build_review(
    scan_root: Path | None = None,
    *,
    pointer: Path | None = None,
) -> dict[str, Any]:
    pointer_path, pointer_payload, pointer_info = current_pointer_payload(pointer)
    pointer_kind = pointer_payload.get("kind") if pointer_payload else None
    pointer_gate = pointer_payload.get("gate") if pointer_payload else None
    if pointer_payload and pointer_kind == "stage5_reentry_recovery_training":
        health_block_reason = reentry_recovery_health_block_reason(pointer_payload)
        if health_block_reason:
            pointer_info.setdefault("preferred", False)
            return {
                "kind": REVIEW_KIND,
                "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "benchmark_assessment": None,
                "benchmark_status": None,
                "benchmark_passed": False,
                "arc_challenge_delta_recurrent_vs_base": None,
                "dense_control_assessment": None,
                "dense_control_status": None,
                "dense_control_passed": False,
                "current_pointer": pointer_info,
                "action": "stop_reentry_recovery_health_needs_review",
                "next_target": "",
                "next_step": health_block_reason + " Do not run debiased benchmark, dense control, or breadth diagnostics yet.",
                "launch_env": {},
            }
    if pointer_payload and pointer_gate not in {BENCHMARK_GATE, DENSE_GATE} and pointer_kind != "stage5_reentry_recovery_training":
        pointer_info.setdefault("preferred", False)
        return {
            "kind": REVIEW_KIND,
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_assessment": None,
            "benchmark_status": None,
            "benchmark_passed": False,
            "arc_challenge_delta_recurrent_vs_base": None,
            "dense_control_assessment": None,
            "dense_control_status": None,
            "dense_control_passed": False,
            "current_pointer": pointer_info,
            "action": "wait_for_reentry_recovery_training",
            "next_target": "",
            "next_step": "Current pointer is before Stage 4 recovery; complete reentry_recovery_training before judging the Phase 1 benchmark gate.",
            "launch_env": {},
        }
    benchmark_path: Path | None
    benchmark_payload: dict[str, Any] | None
    dense_path: Path | None
    dense_payload: dict[str, Any] | None

    if pointer_payload and pointer_payload.get("gate") == BENCHMARK_GATE and pointer_path:
        benchmark_path, benchmark_payload = pointer_path, pointer_payload
        pointer_info["preferred"] = True
    else:
        benchmark_source = pointer_path if pointer_payload and pointer_payload.get("kind") == "stage5_reentry_recovery_training" else None
        benchmark_path, benchmark_payload = latest_payload(
            scan_root=scan_root,
            gate=BENCHMARK_GATE,
            source_summary=benchmark_source,
        )

    if pointer_payload and pointer_payload.get("gate") == DENSE_GATE and pointer_path:
        dense_path, dense_payload = pointer_path, pointer_payload
        pointer_info["preferred"] = True
    else:
        dense_path, dense_payload = latest_payload(scan_root=scan_root, gate=DENSE_GATE)

    pointer_info.setdefault("preferred", False)

    hard_delta = benchmark_hard_delta(benchmark_payload)

    if not benchmark_payload or not benchmark_path:
        action = "wait_for_debiased_benchmark_suite"
        next_target = ""
        next_step = "No broader benchmark assessment found; run debiased_benchmark_suite after Stage 4 recovery passes."
        launch_env: dict[str, str] = {}
    elif not benchmark_passed(benchmark_payload):
        action = "stop_recurrent_not_base_competitive"
        next_target = ""
        next_step = "Recurrent benchmark assessment did not pass base-competitiveness; return to deterministic recovery before dense control or particles."
        launch_env = {}
    elif not dense_payload or not dense_path:
        action = "run_dense_same_curriculum_control"
        next_target = "dense_mcq_trace_sft_control"
        next_step = "Recurrent recovered enough to benchmark; run dense same-curriculum control before claiming architecture lift."
        launch_env = launch_env_for_dense_control(benchmark_path)
    else:
        dense_status = str(dense_payload.get("status") or "")
        dense_passed = bool(dense_payload.get("passed"))
        if dense_passed and dense_status == "hard_tail_lift_vs_dense":
            action = "phase1_architecture_signal"
            next_target = "phase2_breadth_diagnostic_after_review"
            next_step = "Recurrent beats base and dense control on the hard-tail gate; review with strategy agent, replicate on a larger slice, then resume breadth diagnostics."
        elif dense_status == "mixed_hard_tail_signal_vs_dense":
            action = "mixed_architecture_signal_review_surfaces"
            next_target = ""
            next_step = "Some hard-tail surface favors recurrence but evidence is mixed; inspect surface disagreement before breadth/SVGD."
        elif dense_status == "easy_surface_invariance_issue":
            action = "repair_surface_invariance_before_phase2"
            next_target = ""
            next_step = "Surface invariance remains a confound; repair scoring/conditioning before judging architecture lift."
        elif dense_status == "needs_review":
            action = "review_dense_control_metadata"
            next_target = ""
            next_step = "Dense/recurrent control metadata mismatch or insufficient comparability; rerun matched control before judging the architecture."
        else:
            action = "no_architecture_lift_vs_dense"
            next_target = ""
            next_step = "Dense same-curriculum control matches or beats recurrence; improve deterministic depth routing/re-entry before particles or scale claims."
        launch_env = {}

    return {
        "kind": REVIEW_KIND,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_assessment": path_for_cli(benchmark_path) if benchmark_path else None,
        "benchmark_status": benchmark_payload.get("status") if benchmark_payload else None,
        "benchmark_passed": bool(benchmark_payload.get("passed")) if benchmark_payload else False,
        "arc_challenge_delta_recurrent_vs_base": hard_delta,
        "dense_control_assessment": path_for_cli(dense_path) if dense_path else None,
        "dense_control_status": dense_payload.get("status") if dense_payload else None,
        "dense_control_passed": bool(dense_payload.get("passed")) if dense_payload else False,
        "current_pointer": pointer_info,
        "action": action,
        "next_target": next_target,
        "next_step": next_step,
        "launch_env": launch_env,
    }


def report_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Stage 5 Phase 1 Gate Review",
        "",
        f"- Benchmark assessment: `{payload.get('benchmark_assessment') or 'none'}`",
        f"- Benchmark status: `{payload.get('benchmark_status')}`",
        f"- Benchmark passed: `{payload.get('benchmark_passed')}`",
        f"- ARC-Challenge recurrent-vs-base delta: `{payload.get('arc_challenge_delta_recurrent_vs_base')}`",
        f"- Dense control assessment: `{payload.get('dense_control_assessment') or 'none'}`",
        f"- Dense control status: `{payload.get('dense_control_status')}`",
        f"- Dense control passed: `{payload.get('dense_control_passed')}`",
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
                f"- Gate: `{pointer.get('gate')}`",
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
    run_name = run_id or f"stage5_phase1_gate_review_{time.strftime('%Y%m%d_%H%M%S')}"
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
