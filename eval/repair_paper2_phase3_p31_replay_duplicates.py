"""Repair replay duplicates in a resumable Phase 3.1 model-score JSONL.

Interrupted Drive-backed runs can replay the last flushed generation block when
DriveFS catches up after a replacement VM starts.  This command preserves the
raw file, proves that replay rows share immutable lineage, and applies a
first-durable-write rule before normal receipt assembly resumes.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.paper2_phase3_p31_completion import sha256_file


IMMUTABLE_SCORE_KEYS = (
    "kind",
    "model_key",
    "model",
    "revision",
    "battery",
    "battery_role",
    "partition",
    "document_id",
    "content_sha256",
    "item_id",
    "reader",
    "generation_batch_size",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def repair_replay_duplicates(
    *,
    score_path: Path,
    source_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    scores = read_jsonl(score_path)
    source = [
        row
        for row in read_jsonl(source_path)
        if row.get("partition") in {"dev", "verified_train"}
    ]
    source_lookup = {str(row["item_id"]): row for row in source}
    if len(source_lookup) != len(source):
        raise RuntimeError("P3.1 duplicate repair requires unique source item ids")

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in scores:
        groups[str(row["item_id"])].append(row)
    duplicates = {item_id: rows for item_id, rows in groups.items() if len(rows) > 1}
    if not duplicates:
        raise RuntimeError("P3.1 duplicate repair found no replay duplicates")
    if set(groups) != set(source_lookup):
        raise RuntimeError("P3.1 duplicate repair score coverage differs from frozen source")

    exact_groups = 0
    generated_text_conflicts = 0
    prediction_conflicts = 0
    correctness_conflicts = 0
    first_correct_last_incorrect = 0
    first_incorrect_last_correct = 0
    duplicate_batteries: collections.Counter[str] = collections.Counter()
    for item_id, replay_rows in duplicates.items():
        source_row = source_lookup[item_id]
        first = replay_rows[0]
        immutable = tuple(first.get(key) for key in IMMUTABLE_SCORE_KEYS)
        for row in replay_rows:
            if tuple(row.get(key) for key in IMMUTABLE_SCORE_KEYS) != immutable:
                raise RuntimeError(f"P3.1 replay duplicate changed immutable lineage: {item_id}")
            if (
                row.get("battery") != source_row.get("battery")
                or row.get("partition") != source_row.get("partition")
                or row.get("document_id") != source_row.get("document_id")
                or row.get("content_sha256") != source_row.get("content_sha256")
                or row.get("reader") != source_row.get("reader")
            ):
                raise RuntimeError(f"P3.1 replay duplicate differs from frozen source: {item_id}")
        rendered = {json.dumps(row, sort_keys=True, separators=(",", ":")) for row in replay_rows}
        exact_groups += int(len(rendered) == 1)
        generated_text_conflicts += int(len({row.get("generated_text") for row in replay_rows}) > 1)
        prediction_conflicts += int(
            len({json.dumps(row.get("prediction"), sort_keys=True) for row in replay_rows}) > 1
        )
        correctness_conflicts += int(len({bool(row.get("correct")) for row in replay_rows}) > 1)
        first_correct_last_incorrect += int(
            bool(replay_rows[0].get("correct")) and not bool(replay_rows[-1].get("correct"))
        )
        first_incorrect_last_correct += int(
            not bool(replay_rows[0].get("correct")) and bool(replay_rows[-1].get("correct"))
        )
        duplicate_batteries[str(first.get("battery"))] += 1

    original_sha256 = sha256_file(score_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        if sha256_file(archive_path) != original_sha256:
            raise RuntimeError("P3.1 duplicate archive exists with a different hash")
    else:
        shutil.copy2(score_path, archive_path)
    if sha256_file(archive_path) != original_sha256:
        raise RuntimeError("P3.1 duplicate archive hash verification failed")

    seen: set[str] = set()
    repaired = []
    for row in scores:
        item_id = str(row["item_id"])
        if item_id in seen:
            continue
        seen.add(item_id)
        repaired.append(row)
    if len(repaired) != len(source) or seen != set(source_lookup):
        raise RuntimeError("P3.1 first-write repair did not produce exact source coverage")
    write_jsonl_atomic(score_path, repaired)

    return {
        "kind": "paper2_phase3_p31_replay_duplicate_repair_v1",
        "status": "repaired_first_durable_write",
        "selection_rule": "first_file_occurrence_per_item_id",
        "score_path": str(score_path),
        "archive_path": str(archive_path),
        "source_path": str(source_path),
        "original_sha256": original_sha256,
        "archive_sha256": sha256_file(archive_path),
        "repaired_sha256": sha256_file(score_path),
        "source_sha256": sha256_file(source_path),
        "original_rows": len(scores),
        "repaired_rows": len(repaired),
        "unique_source_rows": len(source),
        "duplicate_groups": len(duplicates),
        "duplicate_extra_rows": len(scores) - len(repaired),
        "exact_duplicate_groups": exact_groups,
        "generated_text_conflict_groups": generated_text_conflicts,
        "prediction_conflict_groups": prediction_conflicts,
        "correctness_conflict_groups": correctness_conflicts,
        "first_correct_last_incorrect": first_correct_last_incorrect,
        "first_incorrect_last_correct": first_incorrect_last_correct,
        "duplicate_groups_by_battery": dict(sorted(duplicate_batteries.items())),
        "immutable_lineage_mismatches": 0,
        "confirm_rows": sum(row.get("partition") == "confirm" for row in repaired),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=Path, required=True)
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--archive_path", type=Path, required=True)
    parser.add_argument("--output_receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = repair_replay_duplicates(
        score_path=args.score_path,
        source_path=args.source_path,
        archive_path=args.archive_path,
    )
    write_json(args.output_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
