"""Assess Stage 5 Gate 2 recurrent-particle mechanism evidence.

Gate 2 is the mechanism gate before scaling, reinforcement, or treating
particles as a training signal. It reads a recovery-particle summary and asks
whether the particle setting improves the recovered deterministic recurrent
baseline in a selector-relevant way across replicated seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_GATE2_ASSESSMENT_RUN_ID") or time.strftime(
    "stage5_gate2_assessment_%Y%m%d_%H%M%S"
)


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def source_kind(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("recovery_decision"), dict) and isinstance(payload.get("particle_decision"), dict):
        return "recovery_particle_gate"
    return "unknown"


def recovery_passed(payload: dict[str, Any]) -> bool:
    return bool((payload.get("recovery_decision") or {}).get("passed", False))


def particle_passed(payload: dict[str, Any]) -> bool:
    return bool((payload.get("particle_decision") or {}).get("passed", False))


def particle_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = (payload.get("particle_decision") or {}).get("evidence") or {}
    return evidence if isinstance(evidence, dict) else {}


def variant_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = particle_evidence(payload)
    variants = evidence.get("variants") or {}
    rows: list[dict[str, Any]] = []
    for name, variant in sorted(variants.items()):
        if not isinstance(variant, dict):
            continue
        mean_delta = variant.get("mean_delta_vs_tuned") or variant.get("delta_vs_tuned") or {}
        evaluated = int(variant.get("evaluated_seed_count", 0) or 0)
        non_negative = int(variant.get("non_negative_seed_count", 0) or 0)
        selected_delta = float(mean_delta.get("selected_delta", 0.0) or 0.0)
        best_delta = float(mean_delta.get("best_of_k_delta", 0.0) or 0.0)
        rows.append(
            {
                "variant": str(name),
                "passed": bool(variant.get("passed", False)),
                "evaluated_seed_count": evaluated,
                "non_negative_seed_count": non_negative,
                "majority_non_negative": evaluated > 0 and non_negative > evaluated / 2,
                "selected_delta": selected_delta,
                "best_of_k_delta": best_delta,
                "first_delta": float(mean_delta.get("first_delta", 0.0) or 0.0),
                "valid_rate_delta": float(mean_delta.get("valid_rate_delta", 0.0) or 0.0),
            }
        )
    return rows


def best_variant_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    evidence = particle_evidence(payload)
    best_name = evidence.get("best_replicated_variant")
    rows = variant_rows(payload)
    if best_name:
        for row in rows:
            if row["variant"] == str(best_name):
                return row
    passed_rows = [row for row in rows if row["passed"]]
    if passed_rows:
        return max(passed_rows, key=lambda row: (row["selected_delta"], row["best_of_k_delta"]))
    if rows:
        return max(rows, key=lambda row: (row["selected_delta"], row["best_of_k_delta"]))
    return None


def assess_gate2(
    payload: dict[str, Any],
    *,
    source_summary: str,
    min_seed_count: int = 3,
) -> dict[str, Any]:
    rows = variant_rows(payload)
    best = best_variant_row(payload)
    if not recovery_passed(payload):
        status = "failed"
        reason = "Deterministic recurrent recovery did not pass, so particle value is not interpretable yet."
        next_step = "Return to deterministic recurrent recovery before scaling particles."
    elif best is None:
        status = "failed"
        reason = "No particle variants were found in the recovery-particle summary."
        next_step = "Rerun the recovery-particle gate with at least one particle variant."
    elif not particle_passed(payload) or not best["majority_non_negative"]:
        status = "failed"
        reason = "No particle setting showed replicated non-negative selected and best-of-K lift over recovered recurrent."
        next_step = "Defer particle/SVGD training pressure and continue deterministic recurrent recovery or selector design."
    elif best["evaluated_seed_count"] < min_seed_count:
        status = "needs_more_evidence"
        reason = f"Particle lift is promising but was evaluated on fewer than {min_seed_count} seeds."
        next_step = "Replicate the particle setting across more seeds before treating it as Gate 2 evidence."
    elif best["selected_delta"] > 0 and best["best_of_k_delta"] >= 0:
        status = "passed"
        reason = "A replicated particle setting improved selected-answer accuracy without reducing best-of-K coverage."
        next_step = "Replicate at a larger ARC slice, then consider particle-aware training only if the lift survives."
    elif best["selected_delta"] >= 0 and best["best_of_k_delta"] > 0:
        status = "needs_selector_conversion"
        reason = "Particles improved candidate coverage, but selected-answer accuracy did not improve."
        next_step = "Run selector or verifier work on the particle candidate set before training around this mechanism."
    elif best["best_of_k_delta"] > 0:
        status = "needs_review"
        reason = "Particles improved candidate coverage but harmed selected-answer accuracy."
        next_step = "Inspect candidate and selector failures before promoting the particle setting."
    else:
        status = "failed"
        reason = "Particle evidence did not improve selected-answer accuracy or exact candidate coverage."
        next_step = "Return to deterministic recurrent recovery or particle-design screening."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_gate2_particle_mechanism",
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "next_step": next_step,
        "source_summary": source_summary,
        "source_kind": source_kind(payload),
        "min_seed_count": min_seed_count,
        "recovery_passed": recovery_passed(payload),
        "particle_passed": particle_passed(payload),
        "best_variant": best,
        "variants": rows,
    }


def latest_summary(scan_root: Path) -> Path:
    candidates: list[Path] = []
    for path in scan_root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and source_kind(payload) == "recovery_particle_gate":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No Stage 5 recovery-particle summary found.")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Gate 2 Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source: `{payload['source_summary']}`",
        f"- Source kind: `{payload['source_kind']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Variant Evidence",
        "",
        "| Variant | Passed | Seeds | Non-Negative Seeds | Selected Delta | Best-of-K Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            f"| `{row['variant']}` | `{row['passed']}` | {row['evaluated_seed_count']} | "
            f"{row['non_negative_seed_count']} | {row['selected_delta']} | {row['best_of_k_delta']} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", help="Recovery-particle Stage 5 summary to assess. Defaults to latest.")
    parser.add_argument("--scan_root", default="outputs/stage5")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--min_seed_count", type=int, default=3)
    args = parser.parse_args()

    source = resolve_path(args.summary_json) if args.summary_json else latest_summary(resolve_path(args.scan_root))
    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_dir / "summary.md"
    payload = assess_gate2(
        read_json(source),
        source_summary=path_for_cli(source),
        min_seed_count=args.min_seed_count,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
