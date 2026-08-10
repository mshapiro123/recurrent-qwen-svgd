"""Stage P3.3 labels, audit rows, and fixed instruments without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from training.paper2_phase3_p33_prep import (
    fixed_random_projection,
    prepare_training_rows,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def build(
    *,
    coverage_index: Path,
    output_dir: Path,
    canonical_projection: Path | None,
) -> dict[str, Any]:
    records = read_jsonl(coverage_index)
    staged, audit, receipt = prepare_training_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = output_dir / "p33_staged_labels.jsonl"
    audit_path = output_dir / "p33_audit_slice.jsonl"
    write_jsonl(staged_path, staged)
    write_jsonl(audit_path, audit)
    projection = fixed_random_projection()
    projection_path = output_dir / "p33_fixed_random_projection.pt"
    temporary = projection_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "kind": "paper2_phase3_fixed_random_projection_v1",
            "seed": 20260810,
            "shape": list(projection.shape),
            "matrix": projection,
        },
        temporary,
    )
    temporary.replace(projection_path)
    receipt.update(
        {
            "coverage_index": {
                "path": str(coverage_index),
                "sha256": sha256_file(coverage_index),
                "rows": len(records),
            },
            "staged_labels": {
                "path": str(staged_path),
                "sha256": sha256_file(staged_path),
                "rows": len(staged),
            },
            "audit_slice": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
                "rows": len(audit),
            },
            "fixed_instruments": {
                "random_projection": {
                    "path": str(projection_path),
                    "sha256": sha256_file(projection_path),
                    "shape": list(projection.shape),
                    "orthogonality_max_abs_error": float(
                        (projection @ projection.T - torch.eye(projection.shape[0])).abs().max()
                    ),
                },
                "canonical_projection": (
                    {
                        "path": str(canonical_projection),
                        "sha256": sha256_file(canonical_projection),
                    }
                    if canonical_projection is not None
                    else {"path": None, "sha256": None, "required_before_p33_lock": True}
                ),
            },
            "tier1_observatory": {
                "event_grain": "prompt_by_loop",
                "telemetry": [
                    "bridge_write_ratio_r_b",
                    "gradient_dot_write",
                    "tortuosity",
                    "turning_angle_radians",
                    "fixed_point_residual",
                    "effective_rank",
                    "participation_ratio",
                ],
                "a_state_interventions": [
                    "zero",
                    "norm_matched_random",
                    "cross_example",
                    "stale",
                    "bypass",
                ],
                "paired_from_cached_pre_intervention_state": True,
                "a_state_ratio_clipped": False,
                "state_geometry": {
                    "canonical_scratch_shape": [8, 128],
                    "flattened_width": 1024,
                    "bridge_hidden_width": 896,
                    "raw_and_normalized_states_preserved": True,
                    "random_projection_applies_to": "flattened_canonical_scratch_state",
                    "canonical_projection_is_existing_frozen_canonicalizer": True,
                },
                "analysis_rules": {
                    "prompt_level_bootstrap_only": True,
                    "norm_matched_controls": True,
                    "paired_same_cached_pre_intervention_state": True,
                    "randomness_sources_labeled_separately": True,
                    "instrumentation_rng_precision_kernel_equivalence_required_before_lock": True,
                },
                "training_loop_scaffolding_only": True,
            },
        }
    )
    write_json(output_dir / "summary.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage_index", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--canonical_projection", type=Path)
    args = parser.parse_args()
    result = build(
        coverage_index=args.coverage_index,
        output_dir=args.output_dir,
        canonical_projection=args.canonical_projection,
    )
    if result["optimizer_constructed"] or result["optimizer_steps"]:
        raise RuntimeError("P3.3 build target attempted training")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
