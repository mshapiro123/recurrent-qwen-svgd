"""Freeze disjoint EVAL-D/E, one-pass 7B caches, and own-base features."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import (  # noqa: E402
    atomic_copy,
    cache_partition,
    load_drafter,
    load_teacher,
    read_jsonl,
    summarize_cache,
    validate_teacher_drafter_tokenizer_alignment,
    write_json,
)
from eval.prepare_paper2_dc0_eval_b import prior_document_ids  # noqa: E402
from eval.prepare_paper2_dc1_dev_c import read_prior_eval_ids  # noqa: E402
from training.paper2_dc1 import (  # noqa: E402
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


EVAL_TOKENS = 200_000
PARTITION_SEEDS = {"eval_d": 20260731, "eval_e": 20260732}
FEATURE_SHARD_ROWS = 8


def boundary_layer_indices(
    *, prelude_end: int, recurrent_end: int, layers: int
) -> dict[str, int]:
    if not 0 < prelude_end < recurrent_end < layers:
        raise ValueError("invalid prelude/recurrent/coda boundaries")
    return {
        "post_prelude": int(prelude_end),
        "post_recurrent": int(recurrent_end),
        "post_coda": int(layers),
    }


def public_feature_receipt(
    *,
    partition: str,
    shard_receipts: Sequence[dict[str, Any]],
    layer_indices: dict[str, int],
    positions: int,
) -> dict[str, Any]:
    return {
        "kind": "paper2_phase2_own_base_feature_cache_receipt",
        "partition": partition,
        "positions": int(positions),
        "storage_dtype": "bfloat16",
        "training_target_dtype_policy": "upcast_to_full_float32_before_loss",
        "layer_indices": dict(layer_indices),
        "shards": [
            {
                "sha256": str(item["sha256"]),
                "rows": int(item["rows"]),
            }
            for item in shard_receipts
        ],
        "hidden_values_exposed": False,
    }


def _feature_shard_complete(
    path: Path,
    *,
    partition: str,
    start: int,
    stop: int,
    checkpoint_sha256: str,
    layers: dict[str, int],
) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        payload.get("kind") == "paper2_phase2_own_base_feature_shard"
        and payload.get("partition") == partition
        and int(payload.get("row_start", -1)) == start
        and int(payload.get("row_stop", -1)) == stop
        and payload.get("checkpoint_sha256") == checkpoint_sha256
        and payload.get("layer_indices") == layers
        and len(payload.get("rows") or []) == stop - start
    )


@torch.inference_mode()
def cache_own_base_features(
    *,
    wrapper: Any,
    rows: Sequence[dict[str, Any]],
    partition: str,
    destination: Path,
    checkpoint_sha256: str,
    device: str,
) -> dict[str, Any]:
    layers = boundary_layer_indices(
        prelude_end=wrapper.layer_split.prelude_end,
        recurrent_end=wrapper.layer_split.recurrent_end,
        layers=len(wrapper.qwen.layers),
    )
    destination.mkdir(parents=True, exist_ok=True)
    receipts = []
    positions = 0
    for start in range(0, len(rows), FEATURE_SHARD_ROWS):
        stop = min(len(rows), start + FEATURE_SHARD_ROWS)
        path = destination / f"rows_{start:06d}_{stop:06d}.pt"
        if not _feature_shard_complete(
            path,
            partition=partition,
            start=start,
            stop=stop,
            checkpoint_sha256=checkpoint_sha256,
            layers=layers,
        ):
            payload_rows = []
            for row_index in range(start, stop):
                values = torch.tensor([rows[row_index]["input_ids"]], device=device)
                output = wrapper(
                    input_ids=values,
                    attention_mask=torch.ones_like(values),
                    max_loops=1,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                history = output.hidden_states
                if history is None or len(history) != len(wrapper.qwen.layers) + 1:
                    raise RuntimeError(
                        "own-base cache expected embeddings plus one state per layer"
                    )
                payload_rows.append(
                    {
                        "row_index": row_index,
                        "positions": max(0, values.shape[1] - 1),
                        "features": {
                            name: history[index][0, :-1].to(torch.bfloat16).cpu()
                            for name, index in layers.items()
                        },
                    }
                )
            with tempfile.TemporaryDirectory() as temporary_dir:
                local = Path(temporary_dir) / path.name
                torch.save(
                    {
                        "kind": "paper2_phase2_own_base_feature_shard",
                        "partition": partition,
                        "row_start": start,
                        "row_stop": stop,
                        "checkpoint_sha256": checkpoint_sha256,
                        "layer_indices": layers,
                        "storage_dtype": "bfloat16",
                        "rows": payload_rows,
                    },
                    local,
                )
                atomic_copy(local, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        positions += sum(int(row["positions"]) for row in payload["rows"])
        receipts.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": stop - start}
        )
        print(
            f"phase2_feature_cache partition={partition} rows={stop}/{len(rows)}",
            flush=True,
        )
    return public_feature_receipt(
        partition=partition,
        shard_receipts=receipts,
        layer_indices=layers,
        positions=positions,
    )


def _prepare_partition(
    *,
    partition: str,
    output_path: Path,
    tokenizer: Any,
    excluded_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    if output_path.exists():
        rows = read_jsonl(output_path)
        print(f"phase2_partition_resume partition={partition} path={output_path}", flush=True)
    else:
        per_stratum = EVAL_TOKENS // 2
        general, general_ids = collect_probe_rows(
            iter_fineweb_documents(),
            tokenizer,
            stratum="general",
            token_budget=per_stratum,
            excluded_document_ids=excluded_ids,
        )
        code, _code_ids = collect_probe_rows(
            iter_stack_documents(),
            tokenizer,
            stratum="code",
            token_budget=per_stratum,
            excluded_document_ids=excluded_ids | general_ids,
        )
        rows = sorted(
            [*general, *code],
            key=lambda row: stable_fraction(
                str(row["row_id"]), seed=PARTITION_SEEDS[partition]
            ),
        )
    disjoint = assert_dc1_document_disjoint(
        rows, prior_document_ids=excluded_ids, partition=partition
    )
    receipt = write_jsonl(output_path, rows)
    if receipt["tokens"] != EVAL_TOKENS:
        raise RuntimeError(f"{partition} must contain exactly {EVAL_TOKENS} tokens")
    ids = {str(row["document_id"]) for row in rows}
    manifest_path = output_path.with_name("document_manifest.json")
    write_json(
        manifest_path,
        {
            "kind": f"paper2_phase2_{partition}_private_document_manifest",
            "seed": PARTITION_SEEDS[partition],
            "document_ids": sorted(ids),
            **document_manifest(rows),
            "prior_document_overlap": [],
        },
    )
    return rows, {
        "data": {
            key: receipt[key]
            for key in ("sha256", "rows", "tokens", "documents", "document_id_sha256")
        },
        "private_manifest_sha256": sha256_file(manifest_path),
        "disjointness": disjoint,
    }, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--prior_partition_jsonl", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--private_root", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if sha256_file(checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("EVAL-D/E post-D0 checkpoint SHA-256 mismatch")
    source_manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
    prior_ids = prior_document_ids(source_manifest) | read_prior_eval_ids(
        args.prior_partition_jsonl
    )
    tokenizer = AutoTokenizer.from_pretrained(
        DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION
    )
    private_root = Path(args.private_root)
    partitions: dict[str, list[dict[str, Any]]] = {}
    partition_receipts: dict[str, Any] = {}
    for partition in ("eval_d", "eval_e"):
        data_path = private_root / partition / f"{partition}.jsonl"
        rows, receipt, ids = _prepare_partition(
            partition=partition,
            output_path=data_path,
            tokenizer=tokenizer,
            excluded_ids=prior_ids,
        )
        partitions[partition] = rows
        partition_receipts[partition] = receipt
        prior_ids.update(ids)

    drafter_tokenizer, drafter, resize, original_vocab = load_drafter(
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
        rows_by_partition=partitions,
    )
    teacher_caches = {}
    feature_receipts = {}
    for partition, rows in partitions.items():
        teacher_cache_root = private_root / partition / "teacher_cache"
        teacher_caches[partition] = cache_partition(
            teacher_key="teacher_7b",
            teacher_model=teacher,
            drafter_wrapper=drafter,
            rows=rows,
            partition=partition,
            cache_root=teacher_cache_root,
            selected_full_logits={},
            shared_vocab_size=resize.original_tokenizer_size,
            device=args.device,
        )
        feature_receipts[partition] = cache_own_base_features(
            wrapper=drafter,
            rows=rows,
            partition=partition,
            destination=private_root / partition / "own_base_features",
            checkpoint_sha256=args.expected_checkpoint_sha256,
            device=args.device,
        )
    cache_summary = summarize_cache(
        private_root,
        {"teacher_7b": teacher_caches},
    )
    private_cache_summary = private_root / "teacher_cache_summary.json"
    write_json(
        private_cache_summary,
        {
            "kind": "paper2_phase2_eval_de_teacher_cache",
            "status": "complete_unscored",
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "teacher": TEACHER_7B,
            "teacher_revision": TEACHER_7B_REVISION,
            "tokenizer_alignment": alignment,
            "caches": cache_summary,
            "teacher_passes_per_partition": 1,
            "interpretive_scoring": False,
            "read_once_scoring_spent": False,
        },
    )
    public = {
        "kind": "paper2_phase2_eval_de_freeze",
        "status": "complete_frozen_unscored",
        "source_revisions": source_manifest["dataset_revisions"],
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "partitions": {
            partition: {
                **partition_receipts[partition],
                "teacher_cache_summary_sha256": sha256_file(private_cache_summary),
                "own_base_features": feature_receipts[partition],
                "scores_exposed": False,
                "read_once_scoring_spent": False,
            }
            for partition in ("eval_d", "eval_e")
        },
        "cross_partition_document_overlap": sorted(
            {str(row["document_id"]) for row in partitions["eval_d"]}
            & {str(row["document_id"]) for row in partitions["eval_e"]}
        ),
        "training_started": False,
        "optimizer_steps": 0,
        "implementation_choice": (
            "Cache boundary states after layers 6, 18, and 24 in bfloat16; "
            "future fp32 losses must upcast targets. This preserves multi-layer "
            "targets without precommitting exploration loss weights."
        ),
    }
    if public["cross_partition_document_overlap"]:
        raise RuntimeError("EVAL-D and EVAL-E document sets overlap")
    write_json(args.output_summary, public)
    print(json.dumps(public, indent=2, sort_keys=True))
    del teacher, drafter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
