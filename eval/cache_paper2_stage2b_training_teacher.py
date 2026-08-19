"""Build the resumable pinned-14B top-128 Stage 2B training cache."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from training.paper2_stage2b_data import canonical_jsonl_bytes, read_jsonl
from training.paper2_stage2b_runtime import atomic_json, atomic_torch_save, sha256_file


MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
TOP_K = 128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.shard_rows < 1:
        raise ValueError("shard-rows must be positive")
    rows = read_jsonl(args.rows)
    observed = __import__("hashlib").sha256(canonical_jsonl_bytes(rows)).hexdigest()
    if observed != args.expected_corpus_sha256:
        raise RuntimeError("Stage 2B training corpus SHA mismatch")
    if any(token in str(args.rows).casefold() for token in ("confirm", "eval_e")):
        raise RuntimeError("sealed partition named in Stage 2B training cache input")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.json"
    existing: dict[str, object] = {}
    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            existing.get("kind") == "paper2_stage2b_teacher_cache_index_v1"
            and existing.get("status") == "complete"
            and existing.get("corpus_sha256") == observed
            and existing.get("teacher_revision") == REVISION
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    total_shards = (len(rows) + args.shard_rows - 1) // args.shard_rows
    shards: list[dict[str, object]] = []
    pending: list[tuple[int, int, int, Path]] = []
    for shard_index in range(total_shards):
        start = shard_index * args.shard_rows
        stop = min(len(rows), start + args.shard_rows)
        path = args.output_dir / f"shard_{shard_index:04d}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            valid = (
                payload.get("kind") == "paper2_stage2b_teacher_cache_shard_v1"
                and payload.get("corpus_sha256") == observed
                and payload.get("teacher_revision") == REVISION
                and int(payload.get("start", -1)) == start
                and int(payload.get("stop", -1)) == stop
                and len(payload.get("rows", [])) == stop - start
            )
            if valid:
                shards.append(
                    {
                        "file": path.name,
                        "start": start,
                        "stop": stop,
                        "rows": stop - start,
                        "sha256": sha256_file(path),
                    }
                )
                continue
            path.unlink()
        pending.append((shard_index, start, stop, path))

    model = None
    if pending:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            cache_dir=args.model_cache,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
    started = time.time()
    with torch.inference_mode():
        for shard_index, start, stop, path in pending:
            cached = []
            for batch_start in range(start, stop, args.batch_size):
                batch_stop = min(stop, batch_start + args.batch_size)
                batch_rows = rows[batch_start:batch_stop]
                lengths = [len(row["input_ids"]) for row in batch_rows]
                width = max(lengths)
                input_ids = torch.zeros(
                    (len(batch_rows), width), dtype=torch.long, device="cuda"
                )
                attention = torch.zeros_like(input_ids)
                for local, row in enumerate(batch_rows):
                    length = lengths[local]
                    input_ids[local, :length] = torch.tensor(
                        row["input_ids"], dtype=torch.long, device="cuda"
                    )
                    attention[local, :length] = 1
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
                for local, row in enumerate(batch_rows):
                    row_index = batch_start + local
                    logits = output.logits[local, : lengths[local] - 1]
                    top_logits, top_ids = torch.topk(logits, k=TOP_K, dim=-1)
                    cached.append(
                        {
                            "row_index": row_index,
                            "document_id": row["document_id"],
                            "row_id": row["row_id"],
                            "stratum": row["stratum"],
                            "input_ids": torch.tensor(row["input_ids"], dtype=torch.int32),
                            "teacher_topk_token_ids": top_ids.to(torch.int32).cpu(),
                            "teacher_topk_logits": top_logits.to(torch.bfloat16).cpu(),
                        }
                    )
                del output, input_ids, attention
            shard_sha = atomic_torch_save(
                path,
                {
                    "kind": "paper2_stage2b_teacher_cache_shard_v1",
                    "corpus_sha256": observed,
                    "teacher_model": MODEL,
                    "teacher_revision": REVISION,
                    "top_k": TOP_K,
                    "start": start,
                    "stop": stop,
                    "rows": cached,
                },
            )
            shards.append(
                {
                    "file": path.name,
                    "start": start,
                    "stop": stop,
                    "rows": stop - start,
                    "sha256": shard_sha,
                }
            )
            atomic_json(
                args.output_dir / "progress.json",
                {
                    "kind": "paper2_stage2b_teacher_cache_progress_v1",
                    "status": "running",
                    "completed_shards": len(shards),
                    "total_shards": total_shards,
                    "completed_rows": sum(int(item["rows"]) for item in shards),
                    "elapsed_seconds": time.time() - started,
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "confirm_scored": False,
                    "eval_e_scored": False,
                },
            )
            print(
                f"stage2b_teacher_cache_progress shard={shard_index + 1}/{total_shards} "
                f"rows={stop}/{len(rows)}",
                flush=True,
            )

    shards.sort(key=lambda item: int(item["start"]))
    if [int(item["start"]) for item in shards] != [
        index * args.shard_rows for index in range(total_shards)
    ]:
        raise RuntimeError("Stage 2B teacher shard coverage is not contiguous")
    index = {
        "kind": "paper2_stage2b_teacher_cache_index_v1",
        "status": "complete",
        "corpus_sha256": observed,
        "teacher_model": MODEL,
        "teacher_revision": REVISION,
        "top_k": TOP_K,
        "rows": len(rows),
        "next_token_positions": sum(len(row["input_ids"]) - 1 for row in rows),
        "shard_rows": args.shard_rows,
        "shards": shards,
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "weights_dtype": "bfloat16",
            "attention_backend": "sdpa",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(index_path, index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
