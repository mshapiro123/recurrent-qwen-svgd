"""Benchmark static KV caching against the pinned 7B greedy reader."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/content/tm0_repo")
sys.path.insert(0, str(ROOT))

from eval.eval_paper2_phase3_p31_references import (  # noqa: E402
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
    generate_rows,
)


@torch.inference_mode()
def static_generate(model, tokenizer, rows, batch_size):
    result = {}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [_chat_prompt(tokenizer, _generation_prompt(row)[0]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cache_implementation="static",
        )
        width = encoded["input_ids"].shape[1]
        for row, tokens in zip(batch, generated):
            result[str(row["item_id"])] = tokenizer.decode(
                tokens[width:], skip_special_tokens=True
            )
        print(f"tm0_static_cache_progress rows={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return result


def main() -> int:
    rows = [json.loads(line) for line in Path("/content/tm0_panel.jsonl").read_text().splitlines() if line]
    rows = sorted(
        (row for row in rows if row["battery"] == "gsm8k"),
        key=lambda row: hashlib.sha256(
            f"20260825:static-identity:{row['item_id']}".encode()
        ).hexdigest(),
    )[:32]
    spec = MODEL_SPECS["teacher_7b"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], cache_dir="/content/model-cache"
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"], revision=spec["revision"], cache_dir="/content/model-cache",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    started = time.perf_counter()
    baseline = {
        str(row["item_id"]): text
        for row, text in generate_rows(model, tokenizer, rows, device="cuda", batch_size=8)
    }
    baseline_seconds = time.perf_counter() - started
    # Compile/static-cache setup is intentionally included in this first-pass timing.
    started = time.perf_counter()
    static = static_generate(model, tokenizer, rows, 8)
    static_seconds = time.perf_counter() - started
    changed = [item_id for item_id in baseline if baseline[item_id] != static[item_id]]
    result = {
        "kind": "paper2_tm0_7b_static_cache_identity_v1",
        "rows": len(rows),
        "baseline_seconds": baseline_seconds,
        "static_first_pass_seconds": static_seconds,
        "static_first_pass_speedup": baseline_seconds / static_seconds,
        "generated_text_exact": not changed,
        "changed_rows": len(changed),
        "changed_item_ids": changed,
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    path = Path("/content/tm0_dry/teacher_7b_static_identity.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
