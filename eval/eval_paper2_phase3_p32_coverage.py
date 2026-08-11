"""Build the score-blind P3.2 agreement-stratum coverage surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch

from training.paper2_phase2_matched_alpha import document_partition
from training.paper2_phase3_p32 import canonical_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve(path: str | Path, private_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    markers = ("/private/stage0a/", "/private/full/")
    normalized = str(path).replace("\\", "/")
    for marker in markers:
        if marker in normalized:
            resolved = private_root / normalized.split(marker, 1)[1]
            if resolved.is_file():
                return resolved
    raise FileNotFoundError(f"cannot resolve P3.2 source receipt: {path}")


def _samples(private_root: Path) -> list[dict[str, Any]]:
    path = private_root / "sample_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError(f"empty Stage 0A sample manifest: {path}")
    return rows


def _flatten_teacher_field(payload: dict[str, Any], field: str) -> torch.Tensor:
    """Read both legacy flat fixtures and the real row-grouped teacher shards."""

    if field in payload:
        return payload[field]
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise KeyError(f"teacher cache is missing {field!r} and row groups")
    missing = [index for index, row in enumerate(rows) if field not in row]
    if missing:
        raise KeyError(f"teacher cache row groups missing {field!r}: {missing[:8]}")
    row_indices = torch.cat([row["sample_indices"].long() for row in rows])
    top_indices = payload["sample_indices"].long()
    if not torch.equal(row_indices, top_indices):
        raise RuntimeError("teacher cache row-group sample ordering changed")
    return torch.cat([row[field] for row in rows])


def _training_anchor_mask(
    samples: list[dict[str, Any]], *, source: str
) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
    anchors: dict[int, dict[str, Any]] = {}
    for sample in samples:
        anchor = int(sample["anchor_index"])
        anchors.setdefault(
            anchor,
            {
                "document_id": str(sample["document_id"]),
                "stratum": str(sample["stratum"]),
                "row_id": str(sample["row_id"]),
            },
        )
    ordered = [anchors[index] for index in range(max(anchors) + 1)]
    if source == "old":
        evaluation = document_partition(
            [row["document_id"] for row in ordered],
            evaluation_fraction=0.2,
            seed=20260804,
        )
        mask = ~evaluation
        if int(mask.sum()) != 41_969:
            raise RuntimeError(
                f"old P3.2 train-anchor count changed: observed={int(mask.sum())}"
            )
    elif source == "new":
        mask = torch.ones(len(ordered), dtype=torch.bool)
        if len(ordered) != 140_000:
            raise RuntimeError(
                f"new P3.2 train-anchor count changed: observed={len(ordered)}"
            )
    else:
        raise ValueError("P3.2 source must be old or new")
    return mask, anchors


def _agreement_shard_records(
    *,
    source: str,
    lattice_path: Path,
    teacher_path: Path,
    lattice_receipt: dict[str, Any],
    teacher_receipt: dict[str, Any],
    samples: list[dict[str, Any]],
    train_mask: torch.Tensor,
    anchors: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_files: list[dict[str, str]] = []
    for path, receipt in (
        (lattice_path, lattice_receipt),
        (teacher_path, teacher_receipt),
    ):
        observed = sha256_file(path)
        if observed != receipt["sha256"]:
            raise RuntimeError(f"P3.2 source shard hash mismatch: {path}")
        source_files.append({"path": str(path), "sha256": observed})
    lattice = torch.load(lattice_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    lattice_indices = lattice["sample_indices"].long()
    teacher_indices = teacher["sample_indices"].long()
    if not torch.equal(lattice_indices, teacher_indices):
        raise RuntimeError("P3.2 lattice/teacher sample alignment failed")
    teacher_topk_log_probs = _flatten_teacher_field(teacher, "topk_log_probs")
    if teacher_topk_log_probs.shape[0] != teacher_indices.numel():
        raise RuntimeError("P3.2 teacher top-k row count mismatch")
    if len(lattice["records"]) != int(lattice_indices.numel()):
        raise RuntimeError("P3.2 lattice record count mismatch")
    records: list[dict[str, Any]] = []
    for offset, sample_index in enumerate(lattice_indices.tolist()):
        sample = samples[sample_index]
        anchor = int(sample["anchor_index"])
        if not bool(train_mask[anchor]):
            continue
        lattice_record = lattice["records"][offset]
        greedy = lattice_record["greedy_token_ids"]
        student = int(greedy["student_0p5b"])
        teacher_14b = int(greedy["teacher_14b"])
        teacher_32b = (
            int(greedy["teacher_32b"])
            if greedy.get("teacher_32b") is not None
            else None
        )
        topk_log = teacher_topk_log_probs[offset].float()
        if topk_log.numel() < 2:
            raise RuntimeError("14B confident-agreement margin needs at least top-2")
        margin = float(topk_log[0] - topk_log[1])
        valid = lattice["union_mask"][offset].bool()
        union_ids = lattice["union_ids"][offset][valid].long()
        student_log_probs = lattice["model_candidate_log_probs"]["student_0p5b"][
            offset
        ][valid].float()
        student_matches = torch.where(union_ids.eq(student))[0]
        if student_matches.numel() != 1:
            raise RuntimeError("student greedy token is absent from the sparse union")
        student_top1_probability = float(
            student_log_probs[int(student_matches.item())].exp()
        )
        teacher_top1_probability = float(topk_log[0].exp())
        teacher_count = int(lattice_record.get("teacher_count", 1))
        normalized_agreement = float(
            lattice_record.get("normalized_teacher_agreement", 1.0)
        )
        teacher_js_divergence = (
            0.0
            if teacher_count <= 1
            else (1.0 - normalized_agreement) * math.log(teacher_count)
        )
        document = anchors[anchor]
        record_id = canonical_sha256(
            {
                "source": source,
                "sample_key": str(sample["sample_key"]),
                "document_id": document["document_id"],
            }
        )
        records.append(
            {
                "record_id": record_id,
                "source": source,
                "sample_index": int(sample_index),
                "sample_key": str(sample["sample_key"]),
                "anchor_index": anchor,
                "horizon": int(sample["horizon"]),
                "document_id": document["document_id"],
                "row_id": document["row_id"],
                "stratum": document["stratum"],
                "prediction_position": int(sample["prediction_position"]),
                "student_top1": student,
                "teacher_14b_top1": teacher_14b,
                "teacher_32b_top1": teacher_32b,
                "teachability": float(lattice_record["teachability_student_topk"]),
                "student_top1_probability": student_top1_probability,
                "teacher_14b_top1_probability": teacher_top1_probability,
                "teacher_js_divergence": teacher_js_divergence,
                "confident_agreement_margin": margin,
                "flip_candidate_14b": student != teacher_14b,
                "cascade_covered": teacher_32b is not None,
                "cross_scale_consistent": (
                    teacher_32b is not None and teacher_14b == teacher_32b
                ),
            }
        )
    return records, source_files


def load_agreement_records(
    *,
    source: str,
    summary_path: Path,
    private_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = _samples(private_root)
    train_mask, anchors = _training_anchor_mask(samples, source=source)
    lattice_receipts = list(summary["lattice"]["shards"])
    teacher_receipts = list(summary["model_caches"]["teacher_14b"]["shards"])
    if len(lattice_receipts) != len(teacher_receipts):
        raise RuntimeError("P3.2 lattice and 14B receipt ledgers do not align")
    records: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for lattice_receipt, teacher_receipt in zip(lattice_receipts, teacher_receipts):
        lattice_path = _resolve(lattice_receipt["path"], private_root)
        teacher_path = _resolve(teacher_receipt["path"], private_root)
        shard_records, shard_files = _agreement_shard_records(
            source=source,
            lattice_path=lattice_path,
            teacher_path=teacher_path,
            lattice_receipt=lattice_receipt,
            teacher_receipt=teacher_receipt,
            samples=samples,
            train_mask=train_mask,
            anchors=anchors,
        )
        records.extend(shard_records)
        source_files.extend(shard_files)
    return records, {
        "source": source,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "sample_manifest_sha256": sha256_file(private_root / "sample_manifest.jsonl"),
        "source_files_sha256": canonical_sha256(source_files),
        "train_anchors": int(train_mask.sum()),
        "samples": len(records),
    }


def _copy_shard(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "--archive",
            "--partial",
            "--info=progress2",
            str(source),
            str(destination),
        ],
        check=True,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _valid_resumed_shard(
    *,
    output_path: Path,
    receipt_path: Path,
    source: str,
    shard_index: int,
    lattice_sha256: str,
    teacher_sha256: str,
) -> bool:
    if not output_path.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "source": source,
        "shard_index": shard_index,
        "lattice_sha256": lattice_sha256,
        "teacher_sha256": teacher_sha256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False
    return (
        receipt.get("output_sha256") == sha256_file(output_path)
        and receipt.get("records") == _count_nonempty_lines(output_path)
    )


def load_agreement_records_resumable(
    *,
    source: str,
    summary_path: Path,
    private_root: Path,
    resume_shard_dir: Path,
    scratch_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = _samples(private_root)
    train_mask, anchors = _training_anchor_mask(samples, source=source)
    lattice_receipts = list(summary["lattice"]["shards"])
    teacher_receipts = list(summary["model_caches"]["teacher_14b"]["shards"])
    if len(lattice_receipts) != len(teacher_receipts):
        raise RuntimeError("P3.2 lattice and 14B receipt ledgers do not align")
    source_resume_dir = resume_shard_dir / source
    records: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    for shard_index, (lattice_receipt, teacher_receipt) in enumerate(
        zip(lattice_receipts, teacher_receipts)
    ):
        output_path = source_resume_dir / f"shard_{shard_index:05d}.jsonl"
        receipt_path = source_resume_dir / f"shard_{shard_index:05d}.receipt.json"
        lattice_source = _resolve(lattice_receipt["path"], private_root)
        teacher_source = _resolve(teacher_receipt["path"], private_root)
        lattice_staged = scratch_dir / source / lattice_source.relative_to(private_root)
        teacher_staged = scratch_dir / source / teacher_source.relative_to(private_root)
        expected_files = [
            {"path": str(lattice_staged), "sha256": lattice_receipt["sha256"]},
            {"path": str(teacher_staged), "sha256": teacher_receipt["sha256"]},
        ]
        if _valid_resumed_shard(
            output_path=output_path,
            receipt_path=receipt_path,
            source=source,
            shard_index=shard_index,
            lattice_sha256=lattice_receipt["sha256"],
            teacher_sha256=teacher_receipt["sha256"],
        ):
            shard_records = _read_jsonl(output_path)
            source_files.extend(expected_files)
        else:
            _copy_shard(lattice_source, lattice_staged)
            _copy_shard(teacher_source, teacher_staged)
            shard_records, shard_files = _agreement_shard_records(
                source=source,
                lattice_path=lattice_staged,
                teacher_path=teacher_staged,
                lattice_receipt=lattice_receipt,
                teacher_receipt=teacher_receipt,
                samples=samples,
                train_mask=train_mask,
                anchors=anchors,
            )
            if shard_files != expected_files:
                raise RuntimeError("P3.2 staged shard lineage changed")
            write_jsonl(output_path, shard_records)
            write_json(
                receipt_path,
                {
                    "kind": "paper2_phase3_p32_resumable_shard_v1",
                    "source": source,
                    "shard_index": shard_index,
                    "lattice_sha256": lattice_receipt["sha256"],
                    "teacher_sha256": teacher_receipt["sha256"],
                    "output_sha256": sha256_file(output_path),
                    "records": len(shard_records),
                    "optimizer_steps": 0,
                    "p33_training_authorized": False,
                },
            )
            source_files.extend(shard_files)
            lattice_staged.unlink(missing_ok=True)
            teacher_staged.unlink(missing_ok=True)
        records.extend(shard_records)
        if (
            shard_index == 0
            or (shard_index + 1) % 16 == 0
            or shard_index + 1 == len(lattice_receipts)
        ):
            print(
                "phase3_p32_resumable_progress "
                f"source={source} shard={shard_index + 1}/{len(lattice_receipts)} "
                f"records={len(records)}",
                flush=True,
            )
    return records, {
        "source": source,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "sample_manifest_sha256": sha256_file(private_root / "sample_manifest.jsonl"),
        "source_files_sha256": canonical_sha256(source_files),
        "train_anchors": int(train_mask.sum()),
        "samples": len(records),
        "resumable_shards": len(lattice_receipts),
    }


def coverage_surface(
    records: list[dict[str, Any]],
    *,
    teachability_thresholds: list[float],
    margin_thresholds: list[float],
) -> dict[str, Any]:
    total = len(records)
    flips = [row for row in records if row["flip_candidate_14b"]]
    covered = [row for row in records if row["cascade_covered"]]
    concurrent = [row for row in covered if row["cross_scale_consistent"]]
    rows = []
    for teachability in teachability_thresholds:
        candidates = [
            row
            for row in flips
            if float(row["teachability"]) >= float(teachability)
        ]
        strict = [row for row in candidates if row["cross_scale_consistent"]]
        conflicts = [
            row
            for row in candidates
            if row["cascade_covered"] and not row["cross_scale_consistent"]
        ]
        uncovered = [row for row in candidates if not row["cascade_covered"]]
        rows.append(
            {
                "teachability_threshold": float(teachability),
                "14b_flip_candidates": len(candidates),
                "strict_concurrent_write_candidates": len(strict),
                "cross_scale_conflicts": len(conflicts),
                "targeted_32b_extension_candidates": len(uncovered),
                "strict_fraction_of_14b_candidates": (
                    len(strict) / len(candidates) if candidates else None
                ),
                "by_source": {
                    source: sum(row["source"] == source for row in strict)
                    for source in ("old", "new")
                },
                "by_stratum": {
                    stratum: sum(row["stratum"] == stratum for row in strict)
                    for stratum in ("general", "code")
                },
                "by_horizon": {
                    str(horizon): sum(row["horizon"] == horizon for row in strict)
                    for horizon in (1, 2, 3, 4)
                },
            }
        )
    negative_rows = []
    agreements = [row for row in records if not row["flip_candidate_14b"]]
    for margin in margin_thresholds:
        eligible = [
            row
            for row in agreements
            if float(row["confident_agreement_margin"]) >= float(margin)
        ]
        negative_rows.append(
            {
                "margin_threshold": float(margin),
                "confident_agreement_negatives": len(eligible),
                "14b_only_admissible": sum(
                    not row["cascade_covered"] for row in eligible
                ),
                "cascade_covered": sum(row["cascade_covered"] for row in eligible),
            }
        )
    return {
        "population": {
            "training_anchors": len({(row["source"], row["anchor_index"]) for row in records}),
            "positions": total,
            "teacher_student_14b_disagreements": len(flips),
            "teacher_student_14b_agreements": len(agreements),
            "cascade_covered_positions": len(covered),
            "cross_scale_concurrent_positions": len(concurrent),
            "concurrence_rate_within_cascade_coverage": (
                len(concurrent) / len(covered) if covered else None
            ),
        },
        "strict_write_surface": rows,
        "permissive_negative_surface": negative_rows,
        "distillation_eligible_positions": total,
        "thresholds_selected_for_p33": False,
    }


def build_receipt(
    *,
    sources: list[tuple[str, Path, Path]],
    output_index: Path,
    output_summary: Path,
    teachability_thresholds: list[float],
    margin_thresholds: list[float],
    resume_shard_dir: Path | None = None,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    source_receipts = []
    for source, summary, private in sources:
        if resume_shard_dir is None:
            source_records, receipt = load_agreement_records(
                source=source,
                summary_path=summary,
                private_root=private,
            )
        else:
            if scratch_dir is None:
                raise ValueError("resumable P3.2 coverage requires scratch_dir")
            source_records, receipt = load_agreement_records_resumable(
                source=source,
                summary_path=summary,
                private_root=private,
                resume_shard_dir=resume_shard_dir,
                scratch_dir=scratch_dir,
            )
        records.extend(source_records)
        source_receipts.append(receipt)
    ids = [row["record_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("P3.2 agreement coverage record ids are not unique")
    write_jsonl(output_index, records)
    result = {
        "kind": "paper2_phase3_p32_agreement_coverage_receipt_v1",
        "status": "complete_agreement_coverage_surface_not_final_cache",
        "sources": source_receipts,
        "coverage": coverage_surface(
            records,
            teachability_thresholds=teachability_thresholds,
            margin_thresholds=margin_thresholds,
        ),
        "index": {
            "path": str(output_index),
            "sha256": sha256_file(output_index),
            "records": len(records),
        },
        "assertions": {
            "expected_training_anchor_count": (
                sum(row["train_anchors"] for row in source_receipts) == 181_969
            ),
            "four_positions_per_anchor": len(records) == 181_969 * 4,
            "agreement_semantics_only": True,
            "no_correctness_claims": True,
            "confirm_unscored": True,
            "optimizer_absent": True,
            "training_steps_zero": True,
        },
        "remaining_before_final_p32": [
            "verified-stratum base/14B/32B generation and programmatic verification",
            "oracle direction generation on selected concurrent write positions",
            "document-disjoint linear-decodability forecast",
            "numeric threshold selection in the P3.3 lock",
        ],
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }
    failed = [name for name, passed in result["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"P3.2 agreement coverage assertions failed: {failed}")
    write_json(output_summary, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--output_index", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--resume_shard_dir", type=Path)
    parser.add_argument("--scratch_dir", type=Path)
    parser.add_argument(
        "--teachability_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95],
    )
    parser.add_argument(
        "--margin_thresholds", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0]
    )
    args = parser.parse_args()
    result = build_receipt(
        sources=[
            ("old", args.old_summary, args.old_private),
            ("new", args.new_summary, args.new_private),
        ],
        output_index=args.output_index,
        output_summary=args.output_summary,
        teachability_thresholds=args.teachability_thresholds,
        margin_thresholds=args.margin_thresholds,
        resume_shard_dir=args.resume_shard_dir,
        scratch_dir=args.scratch_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
