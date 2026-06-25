"""Review the Stage 4 recovery curriculum before spending GPU.

This is intentionally CPU-only. It checks whether the trace curriculum that
Stage 4 will consume has enough direct and depth-labeled rows for a bounded
re-entry recovery smoke, and it also calls out when the data is too small for a
benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.reentry_recovery_config import assess_trace_curriculum_for_reentry_recovery

DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
LEGACY_DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd")
DEFAULT_TRACE_COLLECTION = (
    "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json"
)


def resolve_path(path: str | Path, *, root: Path = ROOT) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    return candidate if candidate.is_absolute() else root / candidate


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


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def is_gate_ready_trace_collection(path: Path) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return False
    return (
        payload.get("kind") == "stage5_capability_ladder_trace_collection"
        and payload.get("status") == "trace_curriculum_gate_ready"
        and isinstance(payload.get("gate"), dict)
        and payload["gate"].get("go") is True
    )


def trace_collection_candidates(
    *,
    explicit: str = "",
    root: Path = ROOT,
    extra_roots: tuple[Path, ...] = (DRIVE_ARTIFACT_ROOT, LEGACY_DRIVE_ROOT),
) -> list[Path]:
    explicit = explicit.strip()
    if explicit:
        return [resolve_path(explicit, root=root)]

    candidates: list[Path] = []
    scan_roots = (
        root / "outputs" / "stage5",
        *(extra_root / "outputs" / "stage5" for extra_root in extra_roots),
    )
    for scan_root in scan_roots:
        if scan_root.exists():
            candidates.extend(
                sorted(
                    scan_root.glob("**/summary.json"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
    candidates.append(resolve_path(DEFAULT_TRACE_COLLECTION, root=root))
    return unique_paths(candidates)


def resolve_trace_collection_summary(
    *,
    explicit: str = "",
    root: Path = ROOT,
    extra_roots: tuple[Path, ...] = (DRIVE_ARTIFACT_ROOT, LEGACY_DRIVE_ROOT),
) -> Path:
    for candidate in trace_collection_candidates(
        explicit=explicit,
        root=root,
        extra_roots=extra_roots,
    ):
        if candidate.exists() and is_gate_ready_trace_collection(candidate):
            return candidate
    raise RuntimeError(
        "No gate-ready capability-ladder trace collection summary found. "
        "Set --trace-summary or STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY "
        "to a gate-ready trace collection."
    )


def print_markdown(path: Path, assessment: dict[str, Any]) -> None:
    counts = assessment.get("counts") if isinstance(assessment.get("counts"), dict) else {}
    claim = assessment.get("claim_readiness") if isinstance(assessment.get("claim_readiness"), dict) else {}
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
    print()
    print("## Claim-Sized Readiness")
    print()
    print(f"- Claim-sized go: `{claim.get('go')}`")
    print(f"- Positive row deficit: `{claim.get('positive_row_deficit')}`")
    print(f"- Mode requirements: `{claim.get('mode_requirements')}`")
    print(f"- Target-loop requirements: `{claim.get('target_loop_requirements')}`")
    print(f"- Claim next step: {claim.get('next_step')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-summary",
        default="",
        help=(
            "Explicit trace collection summary. Defaults to "
            "STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY, then "
            "STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY, then latest gate-ready local/Drive summary."
        ),
    )
    parser.add_argument("--min-positive-rows", type=int, default=16)
    parser.add_argument("--claim-min-positive-rows", type=int, default=2000)
    parser.add_argument("--claim-min-mode-rows", default="direct=1000,deep_narrow=1000")
    parser.add_argument("--claim-min-target-loop-rows", default="")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    args = parser.parse_args()

    explicit = (
        args.trace_summary
        or os.environ.get("STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY")
        or os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY")
        or ""
    )
    path = resolve_trace_collection_summary(explicit=explicit)
    assessment = assess_trace_curriculum_for_reentry_recovery(
        read_json(path),
        min_positive_rows=args.min_positive_rows,
        claim_min_positive_rows=args.claim_min_positive_rows,
        claim_min_mode_rows=args.claim_min_mode_rows,
        claim_min_target_loop_rows=args.claim_min_target_loop_rows,
    )
    if args.json:
        print(json.dumps({"trace_summary": path_for_cli(path), **assessment}, indent=2), flush=True)
    else:
        print_markdown(path, assessment)
    return 0 if assessment.get("go") else 1


if __name__ == "__main__":
    raise SystemExit(main())
