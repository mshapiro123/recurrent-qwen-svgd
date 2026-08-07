"""Freeze new Option B documents and emit a locked Stage-0A-compatible cache config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from eval.cache_paper2_phase2_stage0a import write_json, write_jsonl
from training.paper2_phase2_option_b import build_cache_config, load_locked_registration
from training.speculative_depth_d0_corpus import (
    collect_probe_rows,
    iter_fineweb_documents,
    iter_stack_documents,
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collect_excluded_document_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked Option B exclusion source: {path}")
        for row in read_jsonl(path):
            document_id = row.get("document_id")
            if document_id:
                excluded.add(str(document_id))
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output_data", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--anchor_count", required=True, type=int)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--excluded_jsonl", action="append", default=[])
    args = parser.parse_args()

    registration = load_locked_registration(args.registration)
    excluded_paths = [Path(value) for value in args.excluded_jsonl]
    excluded = collect_excluded_document_ids(excluded_paths)
    teacher = registration["teacher_pass"]
    student = teacher["models"]["student_0p5b"]
    tokenizer = AutoTokenizer.from_pretrained(
        student["model"], revision=student["revision"], token=os.environ.get("HF_TOKEN")
    )
    # Six tokens per requested anchor leaves headroom above the four-token
    # non-overlap floor imposed by horizons 1-4.
    token_budget = int(args.anchor_count * 3)
    general_rows, general_documents = collect_probe_rows(
        iter_fineweb_documents(dump=teacher["sources"]["general"]["dump"]),
        tokenizer,
        stratum="general",
        token_budget=token_budget,
        excluded_document_ids=excluded,
    )
    code_rows, code_documents = collect_probe_rows(
        iter_stack_documents(),
        tokenizer,
        stratum="code",
        token_budget=token_budget,
        excluded_document_ids=excluded | general_documents,
    )
    overlap = general_documents & code_documents
    if overlap:
        raise RuntimeError("Option B new-document strata overlap")
    rows = [*general_rows, *code_rows]
    data_receipt = write_jsonl(args.output_data, rows)
    config = build_cache_config(
        registration=registration,
        data_path=args.output_data,
        anchor_count=args.anchor_count,
        run_id=args.run_id,
    )
    write_json(args.output_config, config)
    summary = {
        "kind": "paper2_phase2_option_b_new_document_freeze",
        "status": "complete_training_data_only",
        "anchor_count_requested": args.anchor_count,
        "horizon_sample_count_requested": args.anchor_count * 4,
        "data": data_receipt,
        "excluded_document_count": len(excluded),
        "excluded_document_id_sha256": hashlib.sha256(
            ("\n".join(sorted(excluded)) + "\n").encode("utf-8")
        ).hexdigest(),
        "excluded_sources": [str(path) for path in excluded_paths],
        "new_document_counts": {
            "general": len(general_documents),
            "code": len(code_documents),
        },
        "new_document_overlap": 0,
        "zero_overlap_with_excluded_documents": not bool(
            (general_documents | code_documents) & excluded
        ),
        "new_document_id_sha256": hashlib.sha256(
            ("\n".join(sorted(general_documents | code_documents)) + "\n").encode("utf-8")
        ).hexdigest(),
        "training_started": False,
        "optimizer_steps": 0,
        "evaluation_partition_touched": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
