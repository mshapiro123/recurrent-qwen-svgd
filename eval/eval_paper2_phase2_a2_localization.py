"""Localize A2 writeback help and harm from banked DEV-only row tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from training.paper2_phase2_matched_alpha import document_partition


EXPECTED_ARMS = (
    (0, "full_a2"),
    (0, "draft_only_control"),
    (1, "full_a2"),
    (1, "draft_only_control"),
)
MIN_MASK_ROWS = 200
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260806
OLD_TRAIN_DIAGNOSTIC_SEED = 20260806


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    values = values.float().reshape(-1)
    if not values.numel():
        return {"count": 0}
    quantiles = torch.quantile(
        values, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    )
    return {
        "count": int(values.numel()),
        "minimum": float(values.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _position_bucket(position: int) -> str:
    if position == 0:
        return "token_0"
    if position <= 3:
        return "token_1_3"
    if position <= 31:
        return "token_4_31"
    if position <= 127:
        return "token_32_127"
    return "token_128_plus"


def population_hash_receipt(
    metadata: dict[str, list[Any]],
    *,
    expected_train_anchors: int,
    expected_evaluation_anchors: int,
) -> dict[str, Any]:
    """Hash the banked split and a reproducible, stratum-matched train subset."""

    eval_mask = document_partition(
        metadata["documents"], evaluation_fraction=0.2, seed=20260804
    )
    eval_indices = torch.where(eval_mask)[0].tolist()
    train_indices = torch.where(~eval_mask)[0].tolist()
    if len(train_indices) != expected_train_anchors:
        raise RuntimeError("reconstructed A2 training partition has the wrong size")
    if len(eval_indices) != expected_evaluation_anchors:
        raise RuntimeError("reconstructed A2 evaluation partition has the wrong size")

    def anchor_record(index: int) -> dict[str, Any]:
        return {
            "anchor_index": index,
            "document_id": metadata["documents"][index],
            "stratum": metadata["strata"][index],
            "prediction_position": metadata["positions"][index],
        }

    train_records = [anchor_record(index) for index in train_indices]
    evaluation_records = [anchor_record(index) for index in eval_indices]
    train_documents = {row["document_id"] for row in train_records}
    evaluation_documents = {row["document_id"] for row in evaluation_records}
    overlap = sorted(train_documents & evaluation_documents)
    if overlap:
        raise RuntimeError("A2 train/evaluation document partitions overlap")

    partition = [
        {"document_id": document, "partition": "train"}
        for document in sorted(train_documents)
    ] + [
        {"document_id": document, "partition": "evaluation"}
        for document in sorted(evaluation_documents)
    ]
    partition.sort(key=lambda row: (row["document_id"], row["partition"]))

    target_by_stratum: dict[str, int] = defaultdict(int)
    for row in evaluation_records:
        target_by_stratum[row["stratum"]] += 1
    fixed_subset = []
    for stratum, target in sorted(target_by_stratum.items()):
        candidates = [row for row in train_records if row["stratum"] == stratum]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{OLD_TRAIN_DIAGNOSTIC_SEED}:old_train_diagnostic:"
                    f"{row['document_id']}:{row['anchor_index']}"
                ).encode("utf-8")
            ).hexdigest()
        )
        if len(candidates) < target:
            raise RuntimeError(f"insufficient old-train anchors for stratum {stratum}")
        fixed_subset.extend(candidates[:target])
    fixed_subset.sort(key=lambda row: row["anchor_index"])
    if len(fixed_subset) != expected_evaluation_anchors:
        raise RuntimeError("old-train diagnostic subset has the wrong size")

    exclusion = {
        "train_anchor_count": len(train_records),
        "evaluation_anchor_count": len(evaluation_records),
        "train_document_count": len(train_documents),
        "evaluation_document_count": len(evaluation_documents),
        "overlap_document_count": 0,
        "overlap_documents": [],
    }
    return {
        "algorithm": "document_partition_seed_20260804_then_stratum_matched_hash_rank",
        "old_train_diagnostic_seed": OLD_TRAIN_DIAGNOSTIC_SEED,
        "training_anchor_count": len(train_records),
        "evaluation_anchor_count": len(evaluation_records),
        "fixed_old_train_subset_anchor_count": len(fixed_subset),
        "fixed_old_train_subset_counts_by_stratum": dict(target_by_stratum),
        "existing_training_manifest_sha256": _canonical_sha256(train_records),
        "existing_document_partition_sha256": _canonical_sha256(partition),
        "evaluation_exclusion_sha256": _canonical_sha256(exclusion),
        "fixed_old_train_subset_sha256": _canonical_sha256(fixed_subset),
        "evaluation_manifest_sha256": _canonical_sha256(evaluation_records),
        "exclusion": exclusion,
    }


def load_anchor_metadata(manifest: Path, expected_sha256: str) -> dict[str, list[Any]]:
    if sha256_file(manifest) != expected_sha256:
        raise RuntimeError("Stage 0A sample-manifest hash mismatch")
    anchors: dict[int, tuple[str, str, int]] = {}
    sample_count = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_count += 1
            if int(row["horizon"]) != 1:
                continue
            anchor = int(row["anchor_index"])
            anchors[anchor] = (
                str(row["document_id"]),
                str(row["stratum"]),
                int(row["prediction_position"]),
            )
    if sample_count != 200_000 or len(anchors) != 50_000:
        raise RuntimeError(
            f"unexpected Stage 0A population samples={sample_count} anchors={len(anchors)}"
        )
    if sorted(anchors) != list(range(50_000)):
        raise RuntimeError("Stage 0A anchor indices are not contiguous")
    ordered = [anchors[index] for index in range(50_000)]
    return {
        "documents": [row[0] for row in ordered],
        "strata": [row[1] for row in ordered],
        "positions": [row[2] for row in ordered],
        "position_buckets": [_position_bucket(row[2]) for row in ordered],
    }


def _private_path(receipt_path: str, private_root: Path) -> Path:
    marker = "/private/a2/"
    normalized = receipt_path.replace("\\", "/")
    if marker not in normalized:
        raise RuntimeError(f"unrecognized A2 private receipt path: {receipt_path}")
    return private_root / normalized.split(marker, 1)[1]


def load_arm_rows(
    summary: dict[str, Any], private_root: Path
) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    loaded = {}
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in summary["arms"]}
    if set(by_key) != set(EXPECTED_ARMS):
        raise RuntimeError("A2 summary does not contain the registered four-arm matrix")
    for key in EXPECTED_ARMS:
        receipt = by_key[key]["final_rows"]
        path = _private_path(str(receipt["path"]), private_root)
        if not path.is_file():
            raise FileNotFoundError(f"missing A2 row tensor: {path}")
        if sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"A2 row tensor hash mismatch: {path}")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if int(rows["accepted_length"].numel()) != int(summary["evaluation_anchors"]):
            raise RuntimeError(f"A2 row count mismatch for {key}")
        loaded[key] = rows
    return loaded


def _group_summary(
    mask: torch.Tensor,
    increment: torch.Tensor,
    quality_loss: torch.Tensor,
) -> dict[str, Any]:
    selected = increment[mask]
    selected_loss = quality_loss[mask]
    positive = selected.clamp_min(0)
    negative = (-selected.clamp_max(0))
    return {
        "rows": int(mask.sum()),
        "writeback_increment": _distribution(selected),
        "helped_fraction": float((selected > 0).float().mean()),
        "harmed_fraction": float((selected < 0).float().mean()),
        "positive_mass": float(positive.sum()),
        "negative_mass": float(negative.sum()),
        "quality_loss_rows": int(selected_loss.sum()),
        "quality_loss_fraction": float(selected_loss.float().mean()),
    }


def _document_bootstrap_ci(
    values: torch.Tensor,
    documents: list[str],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for value, document in zip(values.tolist(), documents):
        sums[document] += float(value)
        counts[document] += 1
    keys = sorted(sums)
    document_sums = torch.tensor([sums[key] for key in keys], dtype=torch.float64)
    document_counts = torch.tensor([counts[key] for key in keys], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(replicates):
        selected = torch.randint(len(keys), (len(keys),), generator=generator)
        estimates.append(
            document_sums.index_select(0, selected).sum()
            / document_counts.index_select(0, selected).sum().clamp_min(1)
        )
    stacked = torch.stack(estimates)
    interval = torch.quantile(
        stacked, torch.tensor([0.025, 0.975], dtype=stacked.dtype)
    )
    return float(interval[0]), float(interval[1])


def _candidate_masks(strata: list[str], buckets: list[str]) -> Iterable[tuple[str, torch.Tensor]]:
    for stratum in sorted(set(strata)):
        yield f"stratum={stratum}", torch.tensor([value == stratum for value in strata])
    for bucket in sorted(set(buckets)):
        yield f"position_bucket={bucket}", torch.tensor([value == bucket for value in buckets])
    for stratum in sorted(set(strata)):
        for bucket in sorted(set(buckets)):
            yield (
                f"stratum={stratum};position_bucket={bucket}",
                torch.tensor(
                    [left == stratum and right == bucket for left, right in zip(strata, buckets)]
                ),
            )


def _quantile_groups(values: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    cuts = torch.quantile(values.float(), torch.tensor([0.25, 0.5, 0.75]))
    return [
        ("q1", values <= cuts[0]),
        ("q2", (values > cuts[0]) & (values <= cuts[1])),
        ("q3", (values > cuts[1]) & (values <= cuts[2])),
        ("q4", values > cuts[2]),
    ]


def localize(
    *,
    a2_summary: dict[str, Any],
    metadata: dict[str, list[Any]],
    rows: dict[tuple[int, str], dict[str, torch.Tensor]],
) -> dict[str, Any]:
    eval_mask = document_partition(metadata["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    if int(eval_indices.numel()) != int(a2_summary["evaluation_anchors"]):
        raise RuntimeError("reconstructed A2 evaluation partition has the wrong size")
    documents = [metadata["documents"][index] for index in eval_indices.tolist()]
    strata = [metadata["strata"][index] for index in eval_indices.tolist()]
    buckets = [metadata["position_buckets"][index] for index in eval_indices.tolist()]

    seed_results = []
    increments = {}
    losses = {}
    mask_inputs: dict[int, dict[str, Any]] = {}
    for seed in (0, 1):
        full = rows[(seed, "full_a2")]
        control = rows[(seed, "draft_only_control")]
        if not torch.equal(
            control["bridge_correct_by_horizon"].bool(),
            control["base_correct_by_horizon"].bool(),
        ):
            raise RuntimeError(f"seed {seed} control correctness path is not bit exact")
        increment = full["accepted_length"].float() - control["accepted_length"].float()
        base_correct = full["base_correct_by_horizon"].bool()
        full_correct = full["bridge_correct_by_horizon"].bool()
        control_correct = control["bridge_correct_by_horizon"].bool()
        quality_loss = (base_correct & ~full_correct).any(dim=1)
        increments[seed] = increment
        losses[seed] = quality_loss
        mask_inputs[seed] = {
            "full_accepted": full["accepted_length"].float(),
            "control_accepted": control["accepted_length"].float(),
            "base_correct": base_correct,
            "full_correct": full_correct,
            "control_correct": control_correct,
        }

        structural = {}
        for label, mask in _candidate_masks(strata, buckets):
            structural[label] = _group_summary(mask, increment, quality_loss)
        diagnostics = {}
        diagnostic_values = {
            "probe_kl": full["probe_kl"].float(),
            "probe_top1": full["probe_top1"].float(),
            "draft_gate": full["draft_gate"].float().mean(dim=1),
        }
        for feature, values in diagnostic_values.items():
            diagnostics[feature] = {
                label: _group_summary(mask, increment, quality_loss)
                for label, mask in _quantile_groups(values)
            }
        horizon = []
        for index in range(base_correct.shape[1]):
            horizon.append(
                {
                    "horizon": index + 1,
                    "baseline_correct": int(base_correct[:, index].sum()),
                    "writeback_lost_correct": int(
                        (base_correct[:, index] & ~full_correct[:, index]).sum()
                    ),
                    "writeback_gained_correct": int(
                        (~base_correct[:, index] & full_correct[:, index]).sum()
                    ),
                }
            )
        seed_results.append(
            {
                "seed": seed,
                "overall": _group_summary(torch.ones_like(quality_loss), increment, quality_loss),
                "structural_groups": structural,
                "diagnostic_quantiles": diagnostics,
                "quality_by_horizon": horizon,
            }
        )

    mask_candidates = []
    for candidate_index, (label, mask) in enumerate(_candidate_masks(strata, buckets)):
        per_seed = []
        for seed in (0, 1):
            source = mask_inputs[seed]
            masked_accepted = torch.where(
                mask, source["control_accepted"], source["full_accepted"]
            )
            change = masked_accepted - source["full_accepted"]
            masked_correct = torch.where(
                mask.unsqueeze(1), source["control_correct"], source["full_correct"]
            )
            base_correct = source["base_correct"]
            full_retained = int((source["full_correct"] & base_correct).sum())
            masked_retained = int((masked_correct & base_correct).sum())
            lower, upper = _document_bootstrap_ci(
                change,
                documents,
                seed=BOOTSTRAP_SEED + 100 * candidate_index + seed,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "mean_eal_change_vs_full": float(change.mean()),
                    "document_bootstrap_95_ci": [lower, upper],
                    "retained_correct_change": masked_retained - full_retained,
                    "removed_positive_mass": float(
                        increments[seed][mask].clamp_min(0).sum()
                    ),
                    "removed_negative_mass": float(
                        (-increments[seed][mask].clamp_max(0)).sum()
                    ),
                    "quality_loss_rows_removed": int(losses[seed][mask].sum()),
                }
            )
        qualifies = (
            int(mask.sum()) >= MIN_MASK_ROWS
            and all(row["mean_eal_change_vs_full"] >= 0 for row in per_seed)
            and all(row["document_bootstrap_95_ci"][0] >= 0 for row in per_seed)
            and all(row["retained_correct_change"] >= 0 for row in per_seed)
        )
        mask_candidates.append(
            {
                "label": label,
                "rows": int(mask.sum()),
                "per_seed": per_seed,
                "qualifies_as_structural_mask": qualifies,
            }
        )
    qualified = [row for row in mask_candidates if row["qualifies_as_structural_mask"]]
    qualified.sort(
        key=lambda row: (
            min(seed["document_bootstrap_95_ci"][0] for seed in row["per_seed"]),
            sum(seed["mean_eal_change_vs_full"] for seed in row["per_seed"]),
            row["label"],
        ),
        reverse=True,
    )

    helped_both = (increments[0] > 0) & (increments[1] > 0)
    harmed_both = (increments[0] < 0) & (increments[1] < 0)
    sign_flip = ((increments[0] > 0) & (increments[1] < 0)) | (
        (increments[0] < 0) & (increments[1] > 0)
    )
    return {
        "kind": "paper2_phase2_a2_helped_harmed_localization",
        "status": "complete_cpu_only_banked_rows",
        "model_inference_runs": 0,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "evaluation_anchors": int(eval_indices.numel()),
        "evaluation_documents": len(set(documents)),
        "population_units": {
            "stage0a_anchors": 50_000,
            "stage0a_horizon_samples": 200_000,
            "a2_training_anchors": int(a2_summary["train_anchors"]),
            "a2_evaluation_anchors": int(a2_summary["evaluation_anchors"]),
        },
        "seed_consistency": {
            "helped_both": int(helped_both.sum()),
            "harmed_both": int(harmed_both.sum()),
            "sign_flip": int(sign_flip.sum()),
            "quality_loss_both": int((losses[0] & losses[1]).sum()),
        },
        "seeds": seed_results,
        "mask_decision_rule": {
            "features": ["stratum", "position_bucket", "stratum_x_position_bucket"],
            "minimum_rows_per_seed": MIN_MASK_ROWS,
            "requirements": [
                "nonnegative mean EAL change from replacing full with control in both seeds",
                "document-block bootstrap 95 percent lower bound nonnegative in both seeds",
                "retained baseline-correct horizon decisions do not decrease in either seed",
            ],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "mask_candidates": mask_candidates,
        "recommended_single_mask": qualified[0] if qualified else None,
        "scope": (
            "DEV-only post-hoc localization on banked A2 rows; structural mask is a candidate "
            "for strategy preregistration, not confirmation evidence"
        ),
        "do_not_claim": [
            "diagnostic quantiles are deployable routing features",
            "a post-hoc mask is confirmatory evidence",
            "cached teacher-forced accepted length is serving throughput",
        ],
    }


def run(
    *,
    a2_summary_path: Path,
    stage0a_summary_path: Path,
    stage0a_manifest: Path,
    private_a2: Path,
    output_dir: Path,
) -> dict[str, Any]:
    a2_summary = json.loads(a2_summary_path.read_text(encoding="utf-8"))
    stage0a_summary = json.loads(stage0a_summary_path.read_text(encoding="utf-8"))
    if int(stage0a_summary["manifest"]["anchor_count"]) != 50_000:
        raise RuntimeError("Stage 0A anchor count is not the banked 50,000")
    if int(stage0a_summary["manifest"]["boundary_sample_count"]) != 200_000:
        raise RuntimeError("Stage 0A horizon-sample count is not the banked 200,000")
    metadata = load_anchor_metadata(
        stage0a_manifest, stage0a_summary["manifest"]["sample_manifest_sha256"]
    )
    rows = load_arm_rows(a2_summary, private_a2)
    result = localize(a2_summary=a2_summary, metadata=metadata, rows=rows)
    result["prelock_population_hashes"] = population_hash_receipt(
        metadata,
        expected_train_anchors=int(a2_summary["train_anchors"]),
        expected_evaluation_anchors=int(a2_summary["evaluation_anchors"]),
    )
    result["sources"] = {
        "a2_summary": {
            "path": str(a2_summary_path),
            "sha256": sha256_file(a2_summary_path),
        },
        "stage0a_summary": {
            "path": str(stage0a_summary_path),
            "sha256": sha256_file(stage0a_summary_path),
        },
        "stage0a_manifest": {
            "path": str(stage0a_manifest),
            "sha256": sha256_file(stage0a_manifest),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2_summary", type=Path, required=True)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--stage0a_manifest", type=Path, required=True)
    parser.add_argument("--private_a2", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        a2_summary_path=args.a2_summary,
        stage0a_summary_path=args.stage0a_summary,
        stage0a_manifest=args.stage0a_manifest,
        private_a2=args.private_a2,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "population_units": result["population_units"],
                "seed_consistency": result["seed_consistency"],
                "recommended_single_mask": result["recommended_single_mask"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
