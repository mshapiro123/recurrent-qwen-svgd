"""Assess whether Gate 1 selector/TTA evidence replicated on a second slice.

Gate 1 says a selector or TTA setting improved selected-answer accuracy with
hard-tail support. This script checks the next measurement question: did the
same comparison pass again on a distinct confirmation summary?
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_SELECTOR_REPLICATION_RUN_ID") or time.strftime(
    "stage5_selector_replication_%Y%m%d_%H%M%S"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def gate1_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_gate1_selector_tta" else None


def latest_gate1_summaries(scan_root: Path, *, count: int = 2) -> list[Path]:
    candidates: list[Path] = []
    if not scan_root.exists():
        return []
    for path in scan_root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and gate1_payload(payload):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:count]


def passing_comparisons(payload: dict[str, Any] | None) -> set[str]:
    if not payload:
        return set()
    rows = payload.get("passing_comparisons")
    return {str(item) for item in rows} if isinstance(rows, list) else set()


def comparison_evidence(payload: dict[str, Any] | None, comparison: str) -> dict[str, Any] | None:
    for row in (payload or {}).get("evidence") or []:
        if isinstance(row, dict) and row.get("comparison") == comparison:
            return row
    return None


def gate1_summary(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": path_for_cli(path),
        "present": payload is not None,
        "status": (payload or {}).get("status"),
        "passed": bool((payload or {}).get("passed", False)),
        "source_summary": (payload or {}).get("source_summary"),
        "source_kind": (payload or {}).get("source_kind"),
        "passing_comparisons": sorted(passing_comparisons(payload)),
        "reason": (payload or {}).get("reason"),
        "next_step": (payload or {}).get("next_step"),
    }


def assess_selector_replication(
    *,
    discovery_path: Path | None,
    confirmation_path: Path | None,
    discovery_payload: dict[str, Any] | None,
    confirmation_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    discovery = gate1_payload(discovery_payload or {})
    confirmation = gate1_payload(confirmation_payload or {})
    discovery_passes = passing_comparisons(discovery)
    confirmation_passes = passing_comparisons(confirmation)
    replicated = sorted(discovery_passes & confirmation_passes)

    if discovery is None or confirmation is None:
        status = "needs_confirmation"
        reason = "Two Gate 1 selector/TTA assessment summaries are required for replication evidence."
        next_step = "Run a confirmation Gate 1 assessment on a distinct stratified selector/TTA slice."
    elif not bool(discovery.get("passed")):
        status = "failed"
        reason = "Discovery Gate 1 assessment did not pass."
        next_step = "Do not promote the selector setting; return to recovery or selector design."
    elif not bool(confirmation.get("passed")):
        status = "failed"
        reason = "Confirmation Gate 1 assessment did not pass."
        next_step = "Treat the discovery lift as unreplicated and inspect the confirmation slice."
    elif not replicated:
        status = "failed"
        reason = "Gate 1 passed twice, but no identical selector/TTA comparison passed both slices."
        next_step = "Inspect task mix and selector settings before promoting a specific selector."
    else:
        status = "passed"
        reason = "At least one selector/TTA comparison passed Gate 1 on both discovery and confirmation slices."
        next_step = "Use the replicated selector setting in the next recovered-vs-base benchmark or release gate."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_selector_replication",
        "kind": "selector_replication",
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "next_step": next_step,
        "discovery": gate1_summary(discovery_path, discovery),
        "confirmation": gate1_summary(confirmation_path, confirmation),
        "replicated_comparisons": replicated,
        "replicated_evidence": {
            comparison: {
                "discovery": comparison_evidence(discovery, comparison),
                "confirmation": comparison_evidence(confirmation, comparison),
            }
            for comparison in replicated
        },
    }


def auto_paths(scan_root: Path) -> tuple[Path | None, Path | None]:
    latest = latest_gate1_summaries(scan_root, count=2)
    if len(latest) >= 2:
        return latest[1], latest[0]
    if len(latest) == 1:
        return latest[0], None
    return None, None


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    write_json(output_json, payload)
    lines = [
        f"# Stage 5 Selector Replication - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        f"- Discovery: `{payload['discovery']['path']}` status `{payload['discovery']['status']}`",
        f"- Confirmation: `{payload['confirmation']['path']}` status `{payload['confirmation']['status']}`",
        "",
        "## Replicated Comparisons",
        "",
    ]
    if payload["replicated_comparisons"]:
        for comparison in payload["replicated_comparisons"]:
            lines.append(f"- `{comparison}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This gate does not prove SOTA. It only proves that a selector/TTA "
            "comparison cleared the Stage 5 Gate 1 criteria on two saved slices.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery_gate1_json")
    parser.add_argument("--confirmation_gate1_json")
    parser.add_argument("--scan_root", default="outputs/stage5")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    if args.discovery_gate1_json or args.confirmation_gate1_json:
        discovery_path = resolve_path(args.discovery_gate1_json) if args.discovery_gate1_json else None
        confirmation_path = resolve_path(args.confirmation_gate1_json) if args.confirmation_gate1_json else None
    else:
        discovery_path, confirmation_path = auto_paths(resolve_path(args.scan_root))
    discovery_payload = safe_read_json(discovery_path) if discovery_path else None
    confirmation_payload = safe_read_json(confirmation_path) if confirmation_path else None
    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    payload = assess_selector_replication(
        discovery_path=discovery_path,
        confirmation_path=confirmation_path,
        discovery_payload=discovery_payload,
        confirmation_payload=confirmation_payload,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
