"""Freeze document-disjoint EVAL-B and cache one Qwen2.5-7B teacher pass."""

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

from eval.cache_speculative_depth_d0_teachers import (
    cache_partition,
    load_drafter,
    load_teacher,
    read_jsonl,
    summarize_cache,
    validate_teacher_drafter_tokenizer_alignment,
    write_json,
)
from training.paper2_dc0 import (
    EVAL_B_SEED,
    assert_eval_b_document_disjoint,
    eval_b_document_manifest,
)
from training.speculative_depth_d0_corpus import (
    collect_probe_rows,
    iter_fineweb_documents,
    iter_stack_documents,
    sha256_file,
    stable_fraction,
    write_jsonl,
)
from training.speculative_depth_d0_spec import (
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
    TEACHER_7B,
    TEACHER_7B_REVISION,
)


def prior_document_ids(manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    checked = 0
    for receipt in manifest["artifacts"].values():
        path = Path(str(receipt.get("drive_path") or ""))
        if path.suffix != ".jsonl" or not path.exists():
            continue
        if sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"prior D0 artifact hash mismatch: {path}")
        result.update(str(row["document_id"]) for row in read_jsonl(path))
        checked += 1
    if checked < 5:
        raise RuntimeError("EVAL-B could not verify the prior D0 document universe")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_manifest", required=True)
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

    checkpoint = Path(args.checkpoint)
    if sha256_file(checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("EVAL-B post-D0 checkpoint SHA-256 mismatch")
    manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
    prior_ids = prior_document_ids(manifest)
    output_data = Path(args.output_data)
    if output_data.exists():
        rows = read_jsonl(output_data)
        print(f"eval_b_data_resume={output_data}", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION)
        general, general_ids = collect_probe_rows(
            iter_fineweb_documents(),
            tokenizer,
            stratum="general",
            token_budget=100_000,
            excluded_document_ids=prior_ids,
        )
        code, _code_ids = collect_probe_rows(
            iter_stack_documents(),
            tokenizer,
            stratum="code",
            token_budget=100_000,
            excluded_document_ids=prior_ids | general_ids,
        )
        rows = sorted(
            [*general, *code],
            key=lambda row: stable_fraction(str(row["row_id"]), seed=EVAL_B_SEED),
        )
    disjoint = assert_eval_b_document_disjoint(rows, prior_document_ids=prior_ids)
    receipt = write_jsonl(args.output_data, rows)
    if receipt["tokens"] != 200_000:
        raise RuntimeError("EVAL-B did not freeze exactly 200000 tokens")
    private_document_manifest = output_data.with_name("document_manifest.json")
    document_ids = sorted({str(row["document_id"]) for row in rows})
    write_json(
        private_document_manifest,
        {
            "kind": "paper2_dc0_eval_b_private_document_manifest",
            "seed": EVAL_B_SEED,
            "document_ids": document_ids,
            "document_id_list_sha256": eval_b_document_manifest(rows)["document_id_list_sha256"],
            "prior_document_overlap": [],
        },
    )

    _drafter_tokenizer, drafter, resize, original_vocab = load_drafter(
        checkpoint=checkpoint,
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
        rows_by_partition={"eval_b": rows},
    )
    cache_root = Path(args.private_cache_root)
    cache = cache_partition(
        teacher_key="teacher_7b",
        teacher_model=teacher,
        drafter_wrapper=drafter,
        rows=rows,
        partition="eval_b",
        cache_root=cache_root,
        selected_full_logits={},
        shared_vocab_size=resize.original_tokenizer_size,
        device=args.device,
    )
    caches = summarize_cache(cache_root, {"teacher_7b": {"eval_b": cache}})
    private_summary = {
        "kind": "paper2_dc0_eval_b_teacher_cache",
        "status": "complete",
        "data_jsonl": str(Path(args.output_data)),
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
        "kind": "paper2_dc0_eval_b_freeze",
        "status": "complete_unspent",
        "seed": EVAL_B_SEED,
        "source_revisions": manifest["dataset_revisions"],
        "mix": {"general": 0.5, "code": 0.5},
        "data": {**eval_b_document_manifest(rows), "jsonl_sha256": receipt["sha256"]},
        "private_document_manifest_sha256": sha256_file(private_document_manifest),
        "document_disjointness": disjoint,
        "teacher": {
            "model": TEACHER_7B,
            "revision": TEACHER_7B_REVISION,
            "accepted_positions": caches["teacher_7b"]["eval_b"]["accepted_positions"],
            "rejected_positions": caches["teacher_7b"]["eval_b"]["rejected_positions"],
            "acceptance_rate": caches["teacher_7b"]["eval_b"]["acceptance_rate"],
            "passes": 1,
        },
        "read_log": [
            {
                "purpose": "single-pass teacher cache construction before scoring",
                "interpretive_scoring": False,
            }
        ],
        "read_once_scoring_spent": False,
        "training_started": False,
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
