"""Post-process Stage A prediction caches into oracle and hurt-overlap receipts.

This module never loads a model and never scores EVAL-C again. The public
receipt contains aggregate counts only; position-level predictions remain in
the existing private Drive cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import read_jsonl  # noqa: E402
from eval.eval_speculative_depth_d0_floor import load_partition_cache  # noqa: E402
from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _arm_counts(
    baseline: torch.Tensor, candidate: torch.Tensor, teacher: torch.Tensor
) -> dict[str, int | float]:
    before_correct = baseline.eq(teacher)
    after_correct = candidate.eq(teacher)
    helps = (~before_correct) & after_correct
    hurts = before_correct & (~after_correct)
    scored = int(teacher.numel())
    oracle_correct = int((before_correct | after_correct).sum())
    return {
        "scored_positions": scored,
        "before_correct": int(before_correct.sum()),
        "after_correct": int(after_correct.sum()),
        "helps": int(helps.sum()),
        "hurts": int(hurts.sum()),
        "net": int(helps.sum()) - int(hurts.sum()),
        "oracle_gain": int(helps.sum()),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / scored,
    }


def summarize_oracle_overlap(
    *,
    teacher_rows: Sequence[torch.Tensor],
    baseline_rows: Sequence[torch.Tensor],
    trained_rows: Sequence[torch.Tensor],
    untrained_rows: Sequence[torch.Tensor],
    inplace_rows: Sequence[torch.Tensor],
    strata: Sequence[str],
) -> dict[str, Any]:
    lengths = {
        len(teacher_rows),
        len(baseline_rows),
        len(trained_rows),
        len(untrained_rows),
        len(inplace_rows),
        len(strata),
    }
    if len(lengths) != 1:
        raise ValueError("oracle-overlap inputs must have identical row counts")

    result: dict[str, Any] = {}
    for stratum in (*sorted(set(strata)), "pooled"):
        indices = [
            index
            for index, value in enumerate(strata)
            if stratum == "pooled" or value == stratum
        ]
        if not indices:
            continue
        teacher = torch.cat([teacher_rows[index].long() for index in indices])
        baseline = torch.cat([baseline_rows[index].long() for index in indices])
        trained = torch.cat([trained_rows[index].long() for index in indices])
        untrained = torch.cat([untrained_rows[index].long() for index in indices])
        inplace = torch.cat([inplace_rows[index].long() for index in indices])
        if not (
            teacher.shape
            == baseline.shape
            == trained.shape
            == untrained.shape
            == inplace.shape
        ):
            raise ValueError(f"position alignment failed for stratum {stratum}")

        before_correct = baseline.eq(teacher)
        trained_hurts = before_correct & trained.ne(teacher)
        inplace_hurts = before_correct & inplace.ne(teacher)
        intersection = int((trained_hurts & inplace_hurts).sum())
        union = int((trained_hurts | inplace_hurts).sum())
        trained_count = int(trained_hurts.sum())
        inplace_count = int(inplace_hurts.sum())
        result[stratum] = {
            "scored_positions": int(teacher.numel()),
            "trained_append": _arm_counts(baseline, trained, teacher),
            "untrained_append": _arm_counts(baseline, untrained, teacher),
            "inplace_depth2": _arm_counts(baseline, inplace, teacher),
            "trained_append_vs_inplace_hurts": {
                "trained_hurts": trained_count,
                "inplace_hurts": inplace_count,
                "intersection": intersection,
                "union": union,
                "jaccard": _ratio(intersection, union),
                "trained_contained_in_inplace": _ratio(
                    intersection, trained_count
                ),
                "inplace_contained_in_trained": _ratio(
                    intersection, inplace_count
                ),
            },
        }
    return result


def load_batch_predictions(
    directory: Path,
    *,
    expected_rows: int,
    step: int,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * expected_rows
    paths = sorted(directory.glob("batch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no cached prediction batches under {directory}")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        predictions = payload["predictions"]
        if predictions.ndim != 3 or step >= predictions.shape[-1]:
            raise ValueError(f"unexpected prediction grid in {path}: {predictions.shape}")
        for local, row_index in enumerate(payload["indices"]):
            index = int(row_index)
            if outputs[index] is not None:
                raise RuntimeError(f"duplicate cached row index {index} in {directory}")
            outputs[index] = predictions[local, :, step].long()
    if any(value is None for value in outputs):
        missing = [index for index, value in enumerate(outputs) if value is None]
        raise RuntimeError(f"prediction cache is missing rows: {missing[:20]}")
    return [value for value in outputs if value is not None]


def directory_manifest(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("batch_*.pt"))
    digest = hashlib.sha256()
    for path in paths:
        entry = f"{path.name}:{sha256_file(path)}\n".encode("ascii")
        digest.update(entry)
    return {
        "path": str(directory),
        "batch_files": len(paths),
        "manifest_sha256": digest.hexdigest(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--immutable_cache_receipt", required=True)
    parser.add_argument("--expected_immutable_cache_sha256", required=True)
    parser.add_argument("--inplace_cache_dir", required=True)
    parser.add_argument("--trained_cache_dir", required=True)
    parser.add_argument("--untrained_cache_dir", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--teacher_key", default="teacher_7b")
    parser.add_argument("--partition", default="eval_c")
    args = parser.parse_args()

    immutable_receipt = json.loads(
        Path(args.immutable_cache_receipt).read_text(encoding="utf-8")
    )
    if immutable_receipt.get("sha256") != args.expected_immutable_cache_sha256:
        raise RuntimeError("Stage A immutable scoring-cache receipt SHA mismatch")
    rows = read_jsonl(args.data_jsonl)
    teacher_payload = json.loads(
        Path(args.teacher_cache_summary).read_text(encoding="utf-8")
    )
    teacher_cache = load_partition_cache(
        teacher_payload, args.teacher_key, args.partition
    )
    if len(rows) != len(teacher_cache):
        raise ValueError("data and teacher cache row counts differ")
    teacher_rows = [
        teacher_cache[index]["teacher_greedy_token_id"].long()
        for index in range(len(rows))
    ]
    inplace_dir = Path(args.inplace_cache_dir)
    trained_dir = Path(args.trained_cache_dir)
    untrained_dir = Path(args.untrained_cache_dir)
    baseline = load_batch_predictions(
        inplace_dir, expected_rows=len(rows), step=0
    )
    inplace = load_batch_predictions(
        inplace_dir, expected_rows=len(rows), step=1
    )
    trained = load_batch_predictions(
        trained_dir, expected_rows=len(rows), step=1
    )
    untrained = load_batch_predictions(
        untrained_dir, expected_rows=len(rows), step=1
    )
    summaries = summarize_oracle_overlap(
        teacher_rows=teacher_rows,
        baseline_rows=baseline,
        trained_rows=trained,
        untrained_rows=untrained,
        inplace_rows=inplace,
        strata=[str(row["stratum"]) for row in rows],
    )
    public = {
        "kind": "paper2_phase2_oracle_overlap",
        "status": "complete_exploratory_cache_postprocessing",
        "scoring_reexecuted": False,
        "model_loaded": False,
        "evaluation_c_read_once_scoring_spent": True,
        "source_note": (
            "Position overlap is reconstructed from the private Stage A per-batch "
            "prediction tensors; the immutable aggregate JSONL alone is insufficient."
        ),
        "sources": {
            "data_jsonl_sha256": sha256_file(args.data_jsonl),
            "teacher_cache_summary_sha256": sha256_file(args.teacher_cache_summary),
            "immutable_scoring_cache_sha256": args.expected_immutable_cache_sha256,
            "immutable_scoring_cache_receipt_sha256": sha256_file(
                args.immutable_cache_receipt
            ),
            "inplace_cache": directory_manifest(inplace_dir),
            "trained_cache": directory_manifest(trained_dir),
            "untrained_cache": directory_manifest(untrained_dir),
        },
        "by_stratum": summaries,
        "do_not_claim": [
            "oracle routing is deployable",
            "teacher-token agreement is semantic correctness",
            "the hurt sets are identical unless the overlap statistics establish it",
        ],
    }
    write_json(Path(args.output_summary), public)
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
