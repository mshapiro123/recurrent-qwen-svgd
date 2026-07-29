"""Freeze document-disjoint DEV-C and cache its sole Qwen2.5-7B teacher pass."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import (  # noqa: E402
    cache_partition,
    load_drafter,
    load_teacher,
    read_jsonl,
    summarize_cache,
    validate_teacher_drafter_tokenizer_alignment,
    write_json,
)
from eval.prepare_paper2_dc0_eval_b import prior_document_ids  # noqa: E402
from training.paper2_dc1 import (  # noqa: E402
    DEV_C_SEED,
    DEV_C_TOKENS,
    STRATUM_FRACTIONS,
    assert_dc1_document_disjoint,
    document_manifest,
)
from training.speculative_depth_d0_corpus import (  # noqa: E402
    collect_probe_rows,
    iter_fineweb_documents,
    iter_stack_documents,
    sha256_file,
    stable_fraction,
    write_jsonl,
)
from training.speculative_depth_d0_spec import (  # noqa: E402
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
    TEACHER_7B,
    TEACHER_7B_REVISION,
)


def read_prior_eval_ids(paths: list[str]) -> set[str]:
    ids: set[str] = set()
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"DC1 prior partition is missing: {path}")
        ids.update(str(row["document_id"]) for row in read_jsonl(path))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--prior_partition_jsonl", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--output_data", required=True)
    parser.add_argument("--private_cache_root", required=True)
    parser.add_argument("--private_cache_summary", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("DEV-C post-D0 checkpoint SHA-256 mismatch")
    manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
    prior_ids = prior_document_ids(manifest) | read_prior_eval_ids(args.prior_partition_jsonl)
    output_data = Path(args.output_data)
    if output_data.exists():
        rows = read_jsonl(output_data)
        print(f"dev_c_data_resume={output_data}", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION
        )
        per_stratum = DEV_C_TOKENS // 2
        general, general_ids = collect_probe_rows(
            iter_fineweb_documents(),
            tokenizer,
            stratum="general",
            token_budget=per_stratum,
            excluded_document_ids=prior_ids,
        )
        code, _code_ids = collect_probe_rows(
            iter_stack_documents(),
            tokenizer,
            stratum="code",
            token_budget=per_stratum,
            excluded_document_ids=prior_ids | general_ids,
        )
        rows = sorted(
            [*general, *code],
            key=lambda row: stable_fraction(str(row["row_id"]), seed=DEV_C_SEED),
        )

    disjoint = assert_dc1_document_disjoint(
        rows, prior_document_ids=prior_ids, partition="dev_c"
    )
    receipt = write_jsonl(output_data, rows)
    if receipt["tokens"] != DEV_C_TOKENS:
        raise RuntimeError(f"DEV-C must contain exactly {DEV_C_TOKENS} tokens")
    row_manifest = document_manifest(rows)
    private_manifest = output_data.with_name("document_manifest.json")
    write_json(
        private_manifest,
        {
            "kind": "paper2_dc1_dev_c_private_document_manifest",
            "seed": DEV_C_SEED,
            "document_ids": sorted({str(row["document_id"]) for row in rows}),
            **row_manifest,
            "prior_document_overlap": [],
        },
    )

    _tokenizer, drafter, resize, original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    teacher_tokenizer, teacher = load_teacher(
        model_name=TEACHER_7B,
        revision=TEACHER_7B_REVISION,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    alignment = validate_teacher_drafter_tokenizer_alignment(
        teacher_tokenizer=teacher_tokenizer,
        drafter_original_vocab=original_vocab,
        rows_by_partition={"dev_c": rows},
    )
    cache_root = Path(args.private_cache_root)
    cache = cache_partition(
        teacher_key="teacher_7b",
        teacher_model=teacher,
        drafter_wrapper=drafter,
        rows=rows,
        partition="dev_c",
        cache_root=cache_root,
        selected_full_logits={},
        shared_vocab_size=resize.original_tokenizer_size,
        device=args.device,
    )
    caches = summarize_cache(cache_root, {"teacher_7b": {"dev_c": cache}})
    private_summary = {
        "kind": "paper2_dc1_dev_c_teacher_cache",
        "status": "complete",
        "data_jsonl": str(output_data),
        "data_jsonl_sha256": receipt["sha256"],
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "teacher": TEACHER_7B,
        "teacher_revision": TEACHER_7B_REVISION,
        "tokenizer_alignment": alignment,
        "caches": caches,
        "teacher_passes": 1,
        "teacher_14b_loaded": False,
        "training_started": False,
        "optimizer_steps": 0,
    }
    write_json(args.private_cache_summary, private_summary)
    public = {
        "kind": "paper2_dc1_dev_c_freeze",
        "status": "complete_reusable_development_only",
        "seed": DEV_C_SEED,
        "source_revisions": manifest["dataset_revisions"],
        "mix": dict(STRATUM_FRACTIONS),
        "data": {**row_manifest, **receipt},
        "private_document_manifest_sha256": sha256_file(private_manifest),
        "document_disjointness": disjoint,
        "teacher": {
            "model": TEACHER_7B,
            "revision": TEACHER_7B_REVISION,
            "accepted_positions": caches["teacher_7b"]["dev_c"]["accepted_positions"],
            "rejected_positions": caches["teacher_7b"]["dev_c"]["rejected_positions"],
            "acceptance_rate": caches["teacher_7b"]["dev_c"]["acceptance_rate"],
            "passes": 1,
        },
        "evaluation_c_touched": False,
        "training_started": False,
        "optimizer_steps": 0,
    }
    write_json(args.output_summary, public)
    del teacher, drafter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
