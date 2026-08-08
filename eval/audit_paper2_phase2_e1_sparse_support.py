"""Score-blind sparse-support QC for the frozen Phase-2 E1 EVAL-D cache."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from training.paper2_phase2_e1_confirmation import sha256_file
from training.paper2_phase2_stage0ab import finite_quantiles


MODEL_KEYS = ("student_0p5b", "teacher_7b", "teacher_14b", "teacher_32b")
QC_KIND = "paper2_phase2_e1_sparse_support_qc_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _receipt_lookup(summary: dict[str, Any], model_key: str) -> dict[str, dict[str, Any]]:
    return {
        Path(row["path"]).name: row
        for row in summary["union_scores"][model_key]["shards"]
    }


def audit_sparse_support(
    *, source_summary: Path, private_dir: Path, output_summary: Path
) -> dict[str, Any]:
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    if source.get("status") != "complete_frozen_unscored":
        raise RuntimeError("E1 sparse QC requires the frozen unscored cache")
    required_false = (
        "endpoint_models_loaded",
        "training_started",
        "acceptance_computed",
        "eal_computed",
        "retention_computed",
        "student_teacher_quality_aggregates_emitted",
    )
    if source.get("score_blind") is not True:
        raise RuntimeError("E1 sparse QC source is not score blind")
    for field in required_false:
        if source.get(field) is not False:
            raise RuntimeError(f"E1 sparse QC source contract failed: {field}")
    if int(source.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("E1 sparse QC source has optimizer steps")

    row_metrics: dict[str, dict[str, list[float]]] = {
        key: defaultdict(list) for key in MODEL_KEYS
    }
    counts: dict[str, dict[str, int]] = {
        key: defaultdict(int) for key in MODEL_KEYS
    }
    shard_counts = {key: 0 for key in MODEL_KEYS}
    score_receipts = {key: _receipt_lookup(source, key) for key in MODEL_KEYS}

    for lattice_receipt in source["lattice"]["shards"]:
        filename = Path(lattice_receipt["path"]).name
        union_path = private_dir / "union" / filename
        if not union_path.is_file():
            raise FileNotFoundError(f"missing E1 union shard: {union_path}")
        union = torch.load(union_path, map_location="cpu", weights_only=False)
        union_lookup = {
            int(sample_index): offset
            for offset, sample_index in enumerate(union["sample_indices"].tolist())
        }

        for model_key in MODEL_KEYS:
            receipt = score_receipts[model_key].get(filename)
            if receipt is None:
                raise RuntimeError(f"missing score receipt: {model_key}/{filename}")
            score_path = private_dir / "union_scores" / model_key / filename
            if sha256_file(score_path) != receipt["sha256"]:
                raise RuntimeError(f"score shard hash mismatch: {score_path}")
            score = torch.load(score_path, map_location="cpu", weights_only=False)
            score_lookup = {
                int(sample_index): offset
                for offset, sample_index in enumerate(score["sample_indices"].tolist())
            }
            for audit_offset, sample_index in enumerate(
                score["audit_sample_indices"].tolist()
            ):
                sample_index = int(sample_index)
                union_offset = union_lookup[sample_index]
                score_offset = score_lookup[sample_index]
                mask = union["union_mask"][union_offset].bool()
                ids = union["union_ids"][union_offset][mask].long()
                approximate = score["candidate_log_probs"][score_offset][mask].float()
                reference = score["full_log_probs_bfloat16"][audit_offset].float()
                reference = reference - torch.logsumexp(reference, dim=0)
                reference = reference[ids]

                approximate_finite = torch.isfinite(approximate)
                reference_finite = torch.isfinite(reference)
                both_finite = approximate_finite & reference_finite
                support_mismatch = approximate_finite ^ reference_finite
                mismatch_count = int(support_mismatch.sum())

                counts[model_key]["audit_rows"] += 1
                counts[model_key]["candidate_entries"] += int(mask.sum())
                counts[model_key]["finite_pairs"] += int(both_finite.sum())
                counts[model_key]["support_mismatch_entries"] += mismatch_count
                counts[model_key]["rows_with_support_mismatch"] += int(
                    mismatch_count > 0
                )
                if bool(both_finite.any()):
                    row_metrics[model_key]["finite_log_max_abs_error"].append(
                        float((approximate[both_finite] - reference[both_finite]).abs().max())
                    )
                row_metrics[model_key]["probability_max_abs_error"].append(
                    float((approximate.exp() - reference.exp()).abs().max())
                )
                row_metrics[model_key]["support_mismatch_fraction"].append(
                    float(support_mismatch.float().mean())
                )
                expected_tail = (1.0 - reference.exp().sum()).clamp(1e-30, 1.0)
                actual_tail = score["tail_log_probs"][score_offset].float().exp()
                row_metrics[model_key]["tail_probability_abs_error"].append(
                    float((actual_tail - expected_tail).abs())
                )
                represented_mass = approximate.exp().sum() + actual_tail
                row_metrics[model_key]["represented_mass_abs_error"].append(
                    float((represented_mass - 1.0).abs())
                )
            shard_counts[model_key] += 1

    expected_audit_rows = int(source["manifest"]["full_logit_audit_samples"])
    models: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        model_counts = dict(counts[model_key])
        if model_counts.get("audit_rows") != expected_audit_rows:
            raise RuntimeError(
                f"E1 sparse QC audit-row mismatch for {model_key}: "
                f"{model_counts.get('audit_rows')} != {expected_audit_rows}"
            )
        candidate_entries = int(model_counts["candidate_entries"])
        mismatch_entries = int(model_counts["support_mismatch_entries"])
        model_counts["support_mismatch_fraction_all_entries"] = (
            mismatch_entries / candidate_entries
        )
        summaries = {
            name: finite_quantiles(values)
            for name, values in sorted(row_metrics[model_key].items())
        }
        for name, values in summaries.items():
            if (
                values["positive_infinity_count"]
                or values["negative_infinity_count"]
                or values["nan_count"]
            ):
                raise RuntimeError(f"non-finite E1 sparse QC summary: {model_key}/{name}")
        models[model_key] = {
            "counts": model_counts,
            "metrics": summaries,
            "score_shards_verified": shard_counts[model_key],
        }

    result = {
        "kind": QC_KIND,
        "version": "paper2_phase2_e1_sparse_support_qc_v1_20260808",
        "status": "complete_score_blind_integrity_only",
        "source_summary": str(source_summary),
        "source_summary_sha256": sha256_file(source_summary),
        "private_dir": str(private_dir),
        "models": models,
        "all_emitted_metrics_finite": True,
        "support_mismatch_is_reported_not_silently_dropped": True,
        "interpretation": (
            "Finite-support log errors and explicit support-mismatch counts replace "
            "non-finite legacy log-error quantiles. Probability-space, tail, and "
            "represented-mass errors remain integrity telemetry only."
        ),
        "score_blind": True,
        "endpoint_checkpoints_loaded": False,
        "outcome_scores_computed": False,
        "model_quality_scores_computed": False,
        "read_once_scoring_spent": False,
        "training_started": False,
        "optimizer_steps": 0,
        "ready_for_lock_transcription": True,
        "do_not_claim": [
            "support mismatch is an E1 outcome",
            "sparse reconstruction is exact full-vocabulary scoring",
            "the read-once E1 confirmation has been run",
        ],
    }
    write_json(output_summary, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_summary", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_sparse_support(
        source_summary=args.source_summary,
        private_dir=args.private_dir,
        output_summary=args.output_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
