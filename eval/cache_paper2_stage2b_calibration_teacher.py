"""Cache the pinned 14B top-128 teacher on the frozen Stage 2B calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from training.paper2_stage2b_data import canonical_jsonl_bytes, read_jsonl


MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
TOP_K = 128


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.rows)
    observed_manifest = hashlib.sha256(canonical_jsonl_bytes(rows)).hexdigest()
    if observed_manifest != args.expected_manifest_sha256:
        raise RuntimeError("Stage 2B calibration manifest hash mismatch")
    if args.output.is_file():
        existing = torch.load(args.output, map_location="cpu", weights_only=False)
        if (
            existing.get("kind") == "paper2_stage2b_calibration_teacher_cache_v1"
            and existing.get("manifest_sha256") == observed_manifest
            and existing.get("teacher_revision") == REVISION
        ):
            print(json.dumps(existing["receipt"], indent=2, sort_keys=True))
            return 0

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    cached = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
            output = model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = output.logits[0, :-1].float()
            top_logits, top_ids = torch.topk(logits, k=TOP_K, dim=-1)
            cached.append(
                {
                    "document_id": row["document_id"],
                    "row_id": row["row_id"],
                    "stratum": row["stratum"],
                    "input_ids": torch.tensor(row["input_ids"], dtype=torch.int32),
                    "teacher_topk_token_ids": top_ids.to(torch.int32).cpu(),
                    "teacher_topk_logits": top_logits.to(torch.bfloat16).cpu(),
                }
            )
            print(f"stage2b_teacher_cache_progress row={index + 1}/{len(rows)}", flush=True)
            del output, logits, top_logits, top_ids
    payload = {
        "kind": "paper2_stage2b_calibration_teacher_cache_v1",
        "manifest_sha256": observed_manifest,
        "teacher_model": MODEL,
        "teacher_revision": REVISION,
        "top_k": TOP_K,
        "rows": cached,
    }
    receipt = {
        "kind": "paper2_stage2b_calibration_teacher_cache_receipt_v1",
        "status": "complete_no_student_or_optimizer_contact",
        "manifest_sha256": observed_manifest,
        "teacher_model": MODEL,
        "teacher_revision": REVISION,
        "top_k": TOP_K,
        "rows": len(cached),
        "next_token_positions": sum(item["teacher_topk_token_ids"].shape[0] for item in cached),
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
    payload["receipt"] = receipt
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    receipt["cache_sha256"] = sha256_file(args.output)
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
