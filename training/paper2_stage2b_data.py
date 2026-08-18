"""Deterministic full-sequence corpus contracts for Stage 2B-D."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from training.paper2_phase2_matched_alpha import document_partition


OLD_PARTITION_SEED = 20_260_804
CALIBRATION_SEED = 20_260_818
CALIBRATION_ROWS = 32
MAX_SEQUENCE_LENGTH = 512


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for row in rows
    ).encode("utf-8")


def _stable_rank(*parts: object, seed: int) -> str:
    return hashlib.sha256(
        ":".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")
    ).hexdigest()


def _normalized_row(row: dict[str, Any], *, source_partition: str) -> dict[str, Any]:
    values = [int(value) for value in row.get("input_ids", [])]
    if not 2 <= len(values) <= MAX_SEQUENCE_LENGTH:
        raise ValueError("Stage 2B rows must contain 2..512 token IDs")
    document_id = str(row.get("document_id") or "")
    if not document_id:
        raise ValueError("Stage 2B rows require a document ID")
    stratum = str(row.get("stratum") or "")
    if stratum not in {"general", "code"}:
        raise ValueError(f"unknown Stage 2B training stratum: {stratum}")
    return {
        "document_id": document_id,
        "row_id": str(row.get("row_id") or document_id),
        "stratum": stratum,
        "source_partition": source_partition,
        "input_ids": values,
    }


def build_training_corpus(
    old_rows: Sequence[dict[str, Any]], new_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_documents = [str(row.get("document_id") or "") for row in old_rows]
    old_eval = document_partition(
        old_documents, evaluation_fraction=0.2, seed=OLD_PARTITION_SEED
    )
    selected_old = [
        _normalized_row(row, source_partition="dev_c_train")
        for index, row in enumerate(old_rows)
        if not bool(old_eval[index])
    ]
    selected_new = [
        _normalized_row(row, source_partition="option_b_new_train") for row in new_rows
    ]
    old_ids = {row["document_id"] for row in selected_old}
    new_ids = {row["document_id"] for row in selected_new}
    overlap = old_ids & new_ids
    if overlap:
        raise RuntimeError("Stage 2B old/new training documents overlap")
    rows = [*selected_old, *selected_new]
    payload = canonical_jsonl_bytes(rows)
    by_stratum = Counter(row["stratum"] for row in rows)
    by_source = Counter(row["source_partition"] for row in rows)
    receipt = {
        "kind": "paper2_stage2b_full_sequence_corpus_v1",
        "status": "frozen_before_teacher_or_student_contact",
        "old_document_split": {
            "seed": OLD_PARTITION_SEED,
            "evaluation_fraction": 0.2,
            "selected_training_rows": len(selected_old),
            "excluded_evaluation_rows": len(old_rows) - len(selected_old),
        },
        "new_document_policy": "all Option-B fresh training documents",
        "rows": len(rows),
        "next_token_positions": sum(len(row["input_ids"]) - 1 for row in rows),
        "counts_by_stratum": dict(sorted(by_stratum.items())),
        "counts_by_source": dict(sorted(by_source.items())),
        "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
        "loss_scope": "every non-padding next-token position",
        "corpus_sha256": hashlib.sha256(payload).hexdigest(),
        "document_overlap_old_new": 0,
        "model_contact_before_freeze": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return rows, receipt


def select_calibration_rows(
    rows: Sequence[dict[str, Any]], *, count: int = CALIBRATION_ROWS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 2 or count % 2:
        raise ValueError("Stage 2B calibration count must be an even number at least two")
    per_stratum = count // 2
    selected: list[dict[str, Any]] = []
    for stratum in ("general", "code"):
        candidates = [row for row in rows if row["stratum"] == stratum]
        candidates.sort(
            key=lambda row: _stable_rank(
                row["document_id"], row["row_id"], seed=CALIBRATION_SEED
            )
        )
        if len(candidates) < per_stratum:
            raise ValueError(f"insufficient {stratum} rows for calibration")
        selected.extend(candidates[:per_stratum])
    selected.sort(
        key=lambda row: _stable_rank(
            "ordered", row["document_id"], row["row_id"], seed=CALIBRATION_SEED
        )
    )
    payload = canonical_jsonl_bytes(selected)
    receipt = {
        "kind": "paper2_stage2b_loss_calibration_manifest_v1",
        "status": "frozen_before_model_contact",
        "seed": CALIBRATION_SEED,
        "rows": len(selected),
        "counts_by_stratum": dict(sorted(Counter(row["stratum"] for row in selected).items())),
        "next_token_positions": sum(len(row["input_ids"]) - 1 for row in selected),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "selection": "lowest stable hash within stratum; 1:1 general/code",
        "model_contact_before_freeze": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    return selected, receipt


def write_prelock_packet(
    *, old_data: Path, new_data: Path, output_dir: Path
) -> dict[str, Any]:
    rows, corpus = build_training_corpus(read_jsonl(old_data), read_jsonl(new_data))
    calibration, calibration_receipt = select_calibration_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "training_corpus.jsonl"
    calibration_path = output_dir / "loss_calibration_rows.jsonl"
    corpus_path.write_bytes(canonical_jsonl_bytes(rows))
    calibration_path.write_bytes(canonical_jsonl_bytes(calibration))
    packet = {
        "kind": "paper2_stage2b_data_prelock_packet_v1",
        "status": "complete_no_model_contact",
        "source_data_sha256": {
            "dev_c": sha256_file(old_data),
            "option_b_new": sha256_file(new_data),
        },
        "corpus": corpus,
        "calibration": calibration_receipt,
        "paths": {
            "training_corpus": str(corpus_path),
            "loss_calibration_rows": str(calibration_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet

