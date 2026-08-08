"""Freeze an evaluator-compatible E1 cache without computing an outcome score."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_paper2_phase2_stage0a import read_jsonl, write_json, write_jsonl  # noqa: E402
from training.paper2_phase2_e1_confirmation import REQUIRED_CACHE_FIELDS  # noqa: E402
from training.paper2_phase2_e1_eval_d import (  # noqa: E402
    build_freeze_receipt,
    dev_mixture_weights,
)
from training.paper2_phase2_option_b import build_anchor_admission_rows  # noqa: E402
from training.paper2_phase2_stage0a import sha256_file  # noqa: E402
from training.run_paper2_phase2_matched_alpha import build_pilot_cache  # noqa: E402


def document_ids(path: Path) -> set[str]:
    return {str(row["document_id"]) for row in read_jsonl(path)}


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, partial)
    if sha256_file(source) != sha256_file(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError("E1 private cache copy failed SHA-256 verification")
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--local_cache", type=Path, required=True)
    parser.add_argument("--private_cache", type=Path, required=True)
    parser.add_argument("--data_jsonl", type=Path, required=True)
    parser.add_argument("--data_freeze_summary", type=Path, required=True)
    parser.add_argument("--exclude_jsonl", action="append", type=Path, default=[])
    parser.add_argument("--dev_sample_manifest", type=Path, required=True)
    parser.add_argument("--admission_ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage0a_summary = json.loads(args.stage0a_summary.read_text(encoding="utf-8"))
    if stage0a_summary.get("score_blind") is not True:
        raise RuntimeError("E1 source lattice was not generated in score-blind mode")
    forbidden = (
        "eal_computed",
        "retention_computed",
        "acceptance_computed",
    )
    if any(stage0a_summary.get(key) is not False for key in forbidden):
        raise RuntimeError("E1 source lattice contains a forbidden outcome computation")
    if stage0a_summary.get("student_teacher_quality_aggregates_emitted") is not False:
        raise RuntimeError("E1 source lattice emitted a forbidden quality aggregate")

    data_freeze = json.loads(args.data_freeze_summary.read_text(encoding="utf-8"))
    if data_freeze.get("kind") != "paper2_phase2_e1_eval_d_data_freeze_v1":
        raise RuntimeError("E1 data freeze receipt has the wrong schema")
    if data_freeze.get("status") != "complete_frozen_unscored":
        raise RuntimeError("E1 data partition is not frozen unscored")
    if sha256_file(args.data_jsonl) != data_freeze["data"]["sha256"]:
        raise RuntimeError("E1 data differs from its frozen partition receipt")
    if data_freeze.get("scores_exposed") is not False or data_freeze.get(
        "read_once_scoring_spent"
    ) is not False:
        raise RuntimeError("EVAL-D read-once status is no longer unspent")
    if data_freeze.get("eval_e_touched") is not False:
        raise RuntimeError("EVAL-D materialization unexpectedly touched EVAL-E")

    eval_documents = document_ids(args.data_jsonl)
    overlap: set[str] = set()
    exclusion_receipts = []
    for path in args.exclude_jsonl:
        if not path.is_file():
            raise FileNotFoundError(f"E1 exclusion JSONL is missing: {path}")
        local_overlap = eval_documents & document_ids(path)
        overlap.update(local_overlap)
        exclusion_receipts.append(
            {"path": str(path), "sha256": sha256_file(path), "overlap_count": len(local_overlap)}
        )
    if overlap:
        raise RuntimeError(f"E1 EVAL-D overlaps registered non-evaluation data: {len(overlap)}")

    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.local_cache,
        expected_samples=32_000,
    )
    if set(REQUIRED_CACHE_FIELDS) - set(cache):
        raise RuntimeError("assembled E1 cache is not Option B evaluator-compatible")
    atomic_copy(args.local_cache, args.private_cache)

    samples = read_jsonl(args.stage0a_private / "sample_manifest.jsonl")
    cascade = json.loads(
        (args.stage0a_private / "teacher_32b_cascade_indices.json").read_text(
            encoding="utf-8"
        )
    )
    admission_rows = build_anchor_admission_rows(
        samples, set(int(value) for value in cascade["sample_indices"])
    )
    if len(admission_rows) != 8_000:
        raise RuntimeError("E1 admission ledger does not contain exactly 8,000 anchors")
    if not all(
        len(row["teacher_14b_states_by_horizon"]) == 4
        and all(row["teacher_14b_states_by_horizon"].values())
        for row in admission_rows
    ):
        raise RuntimeError("E1 admission ledger lacks all-anchor 14B states")
    admission_receipt = write_jsonl(args.admission_ledger, admission_rows)

    dev_samples = read_jsonl(args.dev_sample_manifest)
    dev_mixture = dev_mixture_weights(dev_samples)
    model_revisions = {
        key: str(value["revision"])
        for key, value in stage0a_summary["config"]["models"].items()
    }
    receipt = build_freeze_receipt(
        cache=cache,
        private_cache_path=args.private_cache,
        data_sha256=sha256_file(args.data_jsonl),
        document_count=len(eval_documents),
        canonicalizer_sha256=sha256_file(args.canonicalizer),
        sample_manifest_sha256=sha256_file(
            args.stage0a_private / "sample_manifest.jsonl"
        ),
        position_key_sha256_value=stage0a_summary["manifest"]["position_key_sha256"],
        admission_ledger_sha256=admission_receipt["sha256"],
        cascade_count=len(cascade["sample_indices"]),
        dev_mixture=dev_mixture,
        model_revisions=model_revisions,
        cross_partition_document_overlap=sorted(overlap),
    )
    receipt["integrity"] = {
        "eval_d_data_freeze_sha256": sha256_file(args.data_freeze_summary),
        "score_blind_lattice_summary_sha256": sha256_file(args.stage0a_summary),
        "exclusion_receipts": exclusion_receipts,
        "private_admission_ledger_rows": admission_receipt["rows"],
        "private_admission_ledger_sha256": admission_receipt["sha256"],
        "private_cache_transport_sha256_match": (
            sha256_file(args.local_cache) == sha256_file(args.private_cache)
        ),
    }
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
