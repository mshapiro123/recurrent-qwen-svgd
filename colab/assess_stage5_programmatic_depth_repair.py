"""Assess a Stage 5 programmatic direct/deep repair summary.

This is a no-GPU post-run gate. The constructed repair is only useful if it
improves held-out constructed loss without making loop calibration worse. A
passing assessment does not prove benchmark progress; it only justifies running
the ARC routing diagnostic on the produced checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ID = os.environ.get("STAGE5_PROGRAMMATIC_DEPTH_ASSESS_RUN_ID") or (
    f"stage5_programmatic_depth_assessment_{time.strftime('%Y%m%d_%H%M%S')}"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    if not str(path).strip():
        return []
    resolved = resolve_path(path)
    if not resolved.is_file():
        return []
    return [json.loads(line) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metric(payload: dict[str, Any], key: str) -> float | None:
    return finite_float(payload.get(key))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def target_loop_mean(summary: dict[str, Any]) -> float | None:
    rows = read_jsonl(str(summary.get("val_sft") or ""))
    targets = [float(row["target_loop_count"]) for row in rows if isinstance(row.get("target_loop_count"), int)]
    return mean(targets)


def assess(summary: dict[str, Any], *, source_summary: Path) -> dict[str, Any]:
    start_eval = summary.get("start_eval") if isinstance(summary.get("start_eval"), dict) else {}
    best_eval = summary.get("best_eval") if isinstance(summary.get("best_eval"), dict) else {}
    start_loss = metric(start_eval, "loss")
    best_loss = metric(best_eval, "loss")
    start_loops = metric(start_eval, "mean_expected_loops")
    best_loops = metric(best_eval, "mean_expected_loops")
    target_loops = target_loop_mean(summary)

    finite_required = all(value is not None for value in [start_loss, best_loss, start_loops, best_loops])
    loss_delta = None if start_loss is None or best_loss is None else best_loss - start_loss
    if target_loops is not None and start_loops is not None and best_loops is not None:
        start_loop_abs_error = abs(start_loops - target_loops)
        best_loop_abs_error = abs(best_loops - target_loops)
        loop_error_delta = best_loop_abs_error - start_loop_abs_error
    else:
        start_loop_abs_error = None
        best_loop_abs_error = None
        loop_error_delta = None

    loss_improved = loss_delta is not None and loss_delta <= -1e-4
    loop_not_worse = loop_error_delta is None or loop_error_delta <= 0.10
    best_checkpoint = summary.get("best_checkpoint")
    has_checkpoint = bool(best_checkpoint)

    if not finite_required or not has_checkpoint:
        status = "invalid_metrics"
        passed = False
        next_step = "Inspect the programmatic-depth run; metrics or checkpoint path are missing/nonfinite."
    elif loss_improved and loop_not_worse:
        status = "programmatic_depth_lift"
        passed = True
        next_step = "Run ARC routing diagnostic on the programmatic-depth checkpoint."
    elif loss_improved:
        status = "programmatic_depth_lift_loop_warning"
        passed = False
        next_step = "Loss improved but loop calibration worsened; inspect before spending more GPU."
    else:
        status = "programmatic_depth_no_lift"
        passed = False
        next_step = "Do not extend this constructed pass; return to ARC-routing failure analysis."

    return {
        "run_id": RUN_ID,
        "kind": "stage5_programmatic_depth_assessment",
        "source_summary": path_for_cli(source_summary),
        "status": status,
        "passed": passed,
        "checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_checkpoint": {"checkpoint": str(best_checkpoint)} if best_checkpoint else None,
        "evidence": {
            "start_loss": start_loss,
            "best_loss": best_loss,
            "loss_delta": loss_delta,
            "start_mean_expected_loops": start_loops,
            "best_mean_expected_loops": best_loops,
            "target_loop_mean": target_loops,
            "start_loop_abs_error": start_loop_abs_error,
            "best_loop_abs_error": best_loop_abs_error,
            "loop_error_delta": loop_error_delta,
            "loss_improved": loss_improved,
            "loop_not_worse": loop_not_worse,
            "has_checkpoint": has_checkpoint,
        },
        "summary": summary,
        "next_step": next_step,
    }


def write_report(payload: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ev = payload["evidence"]
    lines = [
        f"# Stage 5 Programmatic Depth Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Loss delta: `{ev.get('loss_delta')}`",
        f"- Loop error delta: `{ev.get('loop_error_delta')}`",
        f"- Target loop mean: `{ev.get('target_loop_mean')}`",
        f"- Next step: {payload['next_step']}",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args(argv)

    source_summary = resolve_path(args.summary_json)
    summary = read_json(source_summary)
    output_dir = resolve_path(args.output_dir) if args.output_dir else RUN_DIR
    payload = assess(summary, source_summary=source_summary)
    write_report(payload, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
