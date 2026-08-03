"""Repair non-finite Stage 0A lattice diagnostics from durable cached shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from training.paper2_phase2_stage0a import sha256_file
from training.paper2_phase2_stage0ab import (
    finite_quantiles,
    probability_scale_coherence,
    safe_coarse_lattice_metrics,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _student_topk_mask(union: dict[str, Any], offset: int) -> torch.Tensor:
    union_ids = union["union_ids"][offset].long()
    union_mask = union["union_mask"][offset].bool()
    topk_ids = union["topk_ids"]["student_0p5b"][offset].long()
    result = torch.zeros(union_ids.numel() + 1, dtype=torch.bool)
    valid_ids = union_ids[union_mask]
    positions = torch.searchsorted(valid_ids, topk_ids)
    positions = positions[positions < valid_ids.numel()]
    if positions.numel():
        matched = valid_ids[positions]
        positions = positions[torch.isin(matched, topk_ids)]
        result[positions] = True
    return result


def _distribution(payload: dict[str, Any], model_key: str, offset: int) -> torch.Tensor:
    return torch.cat(
        [
            payload["model_candidate_log_probs"][model_key][offset].float(),
            payload["model_tail_log_probs"][model_key][offset].float().view(1),
        ]
    )


def _repair_audit_shard(
    *, private_dir: Path, filename: str, union: dict[str, Any]
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for model_key in ("student_0p5b", "teacher_7b", "teacher_14b", "teacher_32b"):
        score_path = private_dir / "union_scores" / model_key / filename
        score = torch.load(score_path, map_location="cpu", weights_only=False)
        sample_indices = score["sample_indices"].tolist()
        lookup = {int(value): index for index, value in enumerate(sample_indices)}
        for audit_offset, sample_index in enumerate(score["audit_sample_indices"].tolist()):
            base_offset = lookup[int(sample_index)]
            ids = union["union_ids"][base_offset].long()
            mask = union["union_mask"][base_offset].bool()
            approximate = score["candidate_log_probs"][base_offset][mask].float()
            reference = score["full_log_probs_bfloat16"][audit_offset][ids[mask]].float()
            both_finite = torch.isfinite(approximate) & torch.isfinite(reference)
            support_mismatch = torch.isfinite(approximate) ^ torch.isfinite(reference)
            if bool(both_finite.any()):
                result[f"{model_key}:finite_log_max_abs_error"].append(
                    float((approximate[both_finite] - reference[both_finite]).abs().max())
                )
            result[f"{model_key}:support_mismatch_fraction"].append(
                float(support_mismatch.float().mean())
            )
            result[f"{model_key}:probability_max_abs_error"].append(
                float((approximate.exp() - reference.exp()).abs().max())
            )
    return result


def repair_stage0a(
    *, private_dir: Path, original_summary: Path, repaired_private_dir: Path,
    output_summary: Path
) -> dict[str, Any]:
    source = json.loads(original_summary.read_text(encoding="utf-8"))
    if source.get("status") != "complete_development_only":
        raise RuntimeError("Stage 0A repair requires the completed development receipt")
    if source.get("training_started") or source.get("optimizer_steps"):
        raise RuntimeError("Stage 0A source violated the no-training contract")
    repaired_private_dir.mkdir(parents=True, exist_ok=True)
    aggregates: dict[str, list[float]] = defaultdict(list)
    audit_aggregates: dict[str, list[float]] = defaultdict(list)
    shard_receipts: list[dict[str, Any]] = []
    total = 0
    for shard_number, receipt in enumerate(source["lattice"]["shards"], start=1):
        lattice_path = Path(receipt["path"])
        filename = lattice_path.name
        if sha256_file(lattice_path) != receipt["sha256"]:
            raise RuntimeError(f"Stage 0A lattice shard hash mismatch: {lattice_path}")
        lattice = torch.load(lattice_path, map_location="cpu", weights_only=False)
        union = torch.load(
            private_dir / "union" / filename, map_location="cpu", weights_only=False
        )
        records = []
        for offset, old_record in enumerate(lattice["records"]):
            student = _distribution(lattice, "student_0p5b", offset)
            teachers = [
                _distribution(lattice, "teacher_7b", offset),
                _distribution(lattice, "teacher_14b", offset),
            ]
            if old_record.get("teacher_32b_present"):
                teachers.append(_distribution(lattice, "teacher_32b", offset))
            metric = safe_coarse_lattice_metrics(
                student_log_probs=student,
                teacher_log_probs=teachers,
                student_topk_mask=_student_topk_mask(union, offset),
            )
            coherence = probability_scale_coherence(teachers)
            record = {
                "sample_index": int(old_record["sample_index"]),
                "sample_key": old_record["sample_key"],
                "stratum": old_record["stratum"],
                "horizon": int(old_record["horizon"]),
                "bucket": old_record["bucket"],
                "teacher_32b_present": bool(old_record["teacher_32b_present"]),
                **metric,
                "scale_coherence_probability_cosine": coherence,
            }
            records.append(record)
            for key in (
                "normalized_teacher_agreement",
                "student_gap_coarse_kl_clipped",
                "student_support_miss_mass",
                "teachability_student_topk",
            ):
                aggregates[key].append(float(record[key]))
            if coherence is not None:
                aggregates["scale_coherence_probability_cosine"].append(float(coherence))
        destination = repaired_private_dir / filename
        torch.save(
            {
                "kind": "paper2_phase2_stage0a_repaired_metric_shard",
                "source_lattice_sha256": receipt["sha256"],
                "records": records,
            },
            destination,
        )
        shard_receipts.append(
            {"path": str(destination), "sha256": sha256_file(destination), "samples": len(records)}
        )
        total += len(records)
        local_audit = _repair_audit_shard(
            private_dir=private_dir, filename=filename, union=union
        )
        for key, values in local_audit.items():
            audit_aggregates[key].extend(values)
        print(
            f"stage0a_repair_progress shard={shard_number}/{len(source['lattice']['shards'])} "
            f"samples={total}",
            flush=True,
        )

    if total != int(source["lattice"]["samples"]):
        raise RuntimeError("Stage 0A repaired sample count differs from source")
    repaired = {
        "kind": "paper2_phase2_stage0a_metric_repair",
        "status": "complete_development_only",
        "source_summary": str(original_summary),
        "source_summary_sha256": sha256_file(original_summary),
        "samples": total,
        "metrics": {key: finite_quantiles(values) for key, values in sorted(aggregates.items())},
        "full_logit_audit": {
            key: finite_quantiles(values) for key, values in sorted(audit_aggregates.items())
        },
        "shards": shard_receipts,
        "repair_scope": (
            "finite-support coarse KL with explicit support-miss mass; probability-space "
            "scale coherence; finite-only log-error summaries with support mismatch counts"
        ),
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "do_not_claim": [
            "the clipped coarse KL is an exact full-vocabulary KL",
            "development-only lattice metrics are confirmatory evidence",
            "teacher agreement is correctness",
        ],
    }
    for key, summary in repaired["metrics"].items():
        if summary["nan_count"] or summary["positive_infinity_count"] or summary["negative_infinity_count"]:
            raise RuntimeError(f"Stage 0A repaired metric remains non-finite: {key}")
    write_json(output_summary, repaired)
    return repaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--original_summary", type=Path, required=True)
    parser.add_argument("--repaired_private_dir", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repair_stage0a(
        private_dir=args.private_dir,
        original_summary=args.original_summary,
        repaired_private_dir=args.repaired_private_dir,
        output_summary=args.output_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

