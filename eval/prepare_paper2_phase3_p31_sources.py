"""Materialize the score-blind Phase 3.1 source ledger from pinned datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import load_dataset

from training.paper2_phase3_p31 import (
    ALL_BATTERIES,
    SPLIT_SEED,
    build_split_ledger,
    canonical_sha256,
    partition_rows,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_slice(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0 or size > len(rows):
        raise ValueError(f"invalid fixed slice size {size} for {len(rows)} rows")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:fixed-slice:{row['item_id']}".encode("utf-8")
        ).digest(),
    )
    return ranked[:size]


def gsm8k_final_answer(value: str) -> str:
    matches = re.findall(r"####\s*([^\n]+)", value)
    if not matches:
        raise ValueError("GSM8K answer lacks the registered #### delimiter")
    return matches[-1].strip().replace(",", "")


def arc_rows(dataset: Iterable[Mapping[str, Any]], *, battery: str, native_split: str) -> list[dict[str, Any]]:
    rows = []
    for item in dataset:
        choices = dict(item["choices"])
        prompt = {
            "question": str(item["question"]),
            "choice_labels": [str(value) for value in choices["label"]],
            "choice_text": [str(value) for value in choices["text"]],
        }
        raw_item_id = str(item["id"])
        item_id = f"{battery}-{raw_item_id}"
        rows.append(
            {
                "battery": battery,
                "item_id": item_id,
                "document_id": f"{battery}:{raw_item_id}",
                "native_split": native_split,
                "prompt": prompt,
                "answer": str(item["answerKey"]),
                "tests": None,
                "reader": "cyclic_label_aggregated_permutation_mean_v1",
                "programmatic_verifier_available": native_split == "train",
            }
        )
    return rows


def gsm8k_rows(dataset: Iterable[Mapping[str, Any]], *, native_split: str) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(dataset):
        item_id = f"gsm8k-{native_split}-{index}"
        rows.append(
            {
                "battery": "gsm8k",
                "item_id": item_id,
                "document_id": item_id,
                "native_split": native_split,
                "prompt": str(item["question"]),
                "answer": gsm8k_final_answer(str(item["answer"])),
                "tests": None,
                "reader": "final_number_after_hash_delimiter_v1",
                "programmatic_verifier_available": native_split == "train",
            }
        )
    return rows


def mbpp_rows(dataset: Iterable[Mapping[str, Any]], *, native_split: str) -> list[dict[str, Any]]:
    rows = []
    for item in dataset:
        task_id = str(item["task_id"])
        rows.append(
            {
                "battery": "mbpp",
                "item_id": f"mbpp-{task_id}",
                "document_id": f"mbpp:{task_id}",
                "native_split": native_split,
                "prompt": str(item["prompt"]),
                "answer": str(item["code"]),
                "tests": [*map(str, item["test_imports"]), *map(str, item["test_list"])],
                "reader": "isolated_subprocess_unit_test_execution_v1",
                "programmatic_verifier_available": native_split == "train",
            }
        )
    return rows


def mmlu_rows(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(dataset):
        subject = str(item["subject"])
        item_id = f"mmlu-{subject}-{index}"
        rows.append(
            {
                "battery": "mmlu",
                "item_id": item_id,
                "document_id": item_id,
                "native_split": "evaluation",
                "prompt": {
                    "question": str(item["question"]),
                    "choices": [str(value) for value in item["choices"]],
                },
                "answer": int(item["answer"]),
                "tests": None,
                "reader": "cyclic_label_aggregated_permutation_mean_v1",
                "programmatic_verifier_available": False,
            }
        )
    return rows


def tier1_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item["id"])
            rows.append(
                {
                    "battery": "tier1",
                    "item_id": item_id,
                    "document_id": f"tier1:{item_id}",
                    "native_split": "evaluation",
                    "prompt": str(item["question"]),
                    "answer": str(item["answer_text"]),
                    "tests": None,
                    "reader": "paper_one_same_reader_v1",
                    "programmatic_verifier_available": False,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def materialize(
    source_manifest: Mapping[str, Any],
    *,
    tier1_path: Path,
    private_rows: Path,
    mmlu_slice_size: int,
) -> dict[str, Any]:
    specs = source_manifest["sources"]
    rows: list[dict[str, Any]] = []

    for battery in ("arc_easy", "arc_challenge"):
        spec = specs[battery]
        for source_split, native_split in (
            (spec["evaluation_split"], "evaluation"),
            (spec["verified_train_split"], "train"),
        ):
            if native_split == "train" and battery == "arc_easy":
                continue
            dataset = load_dataset(
                spec["dataset_id"], spec["config"], split=source_split, revision=spec["revision"]
            )
            rows.extend(arc_rows(dataset, battery=battery, native_split=native_split))

    gsm = specs["gsm8k"]
    rows.extend(
        gsm8k_rows(
            load_dataset(
                gsm["dataset_id"], gsm["config"], split=gsm["verified_train_split"], revision=gsm["revision"]
            ),
            native_split="train",
        )
    )
    rows.extend(
        gsm8k_rows(
            load_dataset(
                gsm["dataset_id"], gsm["config"], split=gsm["evaluation_split"], revision=gsm["revision"]
            ),
            native_split="evaluation",
        )
    )

    mbpp = specs["mbpp"]
    rows.extend(
        mbpp_rows(
            load_dataset(
                mbpp["dataset_id"], mbpp["config"], split=mbpp["verified_train_split"], revision=mbpp["revision"]
            ),
            native_split="train",
        )
    )
    rows.extend(
        mbpp_rows(
            load_dataset(
                mbpp["dataset_id"], mbpp["config"], split=mbpp["evaluation_split"], revision=mbpp["revision"]
            ),
            native_split="evaluation",
        )
    )

    mmlu = specs["mmlu"]
    all_mmlu = mmlu_rows(
        load_dataset(
            mmlu["dataset_id"], mmlu["config"], split=mmlu["evaluation_split"], revision=mmlu["revision"]
        )
    )
    selected_mmlu = stable_slice(all_mmlu, size=mmlu_slice_size, seed=SPLIT_SEED)
    rows.extend(selected_mmlu)

    tier1_spec = specs["tier1"]
    if sha256_file(tier1_path) != tier1_spec["sha256"]:
        raise RuntimeError("Tier-1 source SHA mismatch")
    rows.extend(tier1_rows(tier1_path))

    rows.sort(key=lambda row: (row["battery"], row["native_split"], row["item_id"]))
    partitioned_rows = partition_rows(rows)
    write_jsonl(private_rows, partitioned_rows)
    revisions = {battery: str(specs[battery].get("revision") or specs[battery]["sha256"]) for battery in ALL_BATTERIES}
    reader_versions = {
        battery: next(row["reader"] for row in rows if row["battery"] == battery)
        for battery in ALL_BATTERIES
    }
    ledger = build_split_ledger(
        partitioned_rows,
        dataset_revisions=revisions,
        reader_versions=reader_versions,
    )
    return {
        "kind": "paper2_phase3_p31_materialized_source_receipt_v1",
        "status": "score_blind_sources_materialized_confirm_unscored",
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "reader_source_path": str(Path(__file__).resolve()),
        "reader_source_sha256": sha256_file(Path(__file__).resolve()),
        "private_rows_path": str(private_rows),
        "private_rows_sha256": sha256_file(private_rows),
        "mmlu_slice_size": mmlu_slice_size,
        "mmlu_slice_membership_sha256": canonical_sha256(
            [row["item_id"] for row in selected_mmlu]
        ),
        "ledger": ledger,
        "scores_computed": False,
        "models_loaded": False,
        "confirm_scoring_spent": False,
        "training_started": False,
        "optimizer_constructed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--tier1_path", type=Path, required=True)
    parser.add_argument("--private_rows", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--mmlu_slice_size", type=int, default=512)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    result = materialize(
        manifest,
        tier1_path=args.tier1_path,
        private_rows=args.private_rows,
        mmlu_slice_size=args.mmlu_slice_size,
    )
    write_json(args.output_summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
