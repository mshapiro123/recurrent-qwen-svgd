"""Review the Stage 4 recovery curriculum before spending GPU.

This is intentionally CPU-only. It checks whether the trace curriculum that
Stage 4 will consume has enough direct and depth-labeled rows for a bounded
re-entry recovery smoke, and it also calls out when the data is too small for a
benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.reentry_recovery_config import assess_trace_curriculum_for_reentry_recovery

DEFAULT_TRACE_COLLECTION = (
    "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json"
)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    return candidate if candidate.is_absolute() else ROOT / candidate


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def print_markdown(path: Path, assessment: dict[str, Any]) -> None:
    counts = assessment.get("counts") if isinstance(assessment.get("counts"), dict) else {}
    print("# Stage 4 Recovery Curriculum Readiness")
    print()
    print(f"- Trace summary: `{path_for_cli(path)}`")
    print(f"- Status: `{assessment.get('status')}`")
    print(f"- Go: `{assessment.get('go')}`")
    print(f"- Positive rows: `{counts.get('positive_rows')}`")
    print(f"- Mode counts: `{counts.get('mode_counts')}`")
    print(f"- Target loop counts: `{counts.get('target_loop_counts')}`")
    print(f"- Strict mode gate: `{assessment.get('strict_mode_gate')}`")
    print(f"- Strict target-loop gate: `{assessment.get('strict_target_loop_gate')}`")
    print(f"- Warnings: `{assessment.get('warnings')}`")
    print(f"- Issues: `{assessment.get('issues')}`")
    print(f"- Next step: {assessment.get('next_step')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-summary", default=DEFAULT_TRACE_COLLECTION)
    parser.add_argument("--min-positive-rows", type=int, default=16)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    args = parser.parse_args()

    path = resolve_path(args.trace_summary)
    assessment = assess_trace_curriculum_for_reentry_recovery(
        read_json(path),
        min_positive_rows=args.min_positive_rows,
    )
    if args.json:
        print(json.dumps({"trace_summary": path_for_cli(path), **assessment}, indent=2), flush=True)
    else:
        print_markdown(path, assessment)
    return 0 if assessment.get("go") else 1


if __name__ == "__main__":
    raise SystemExit(main())
