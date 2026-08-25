"""Check whether a wider 7B scoring batch is output-identical and cheaper."""

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

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS, generate_rows  # noqa: E402


def main() -> int:
    rows = [
        json.loads(line)
        for line in Path("/content/tm0_panel.jsonl").read_text().splitlines()
        if line
    ]
    rows = sorted(
        (row for row in rows if row["battery"] == "gsm8k"),
        key=lambda row: hashlib.sha256(
            f"20260825:batch-identity:{row['item_id']}".encode()
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
    reads = {}
    timings = {}
    for batch_size in (8, 16):
        started = time.perf_counter()
        reads[batch_size] = {
            str(row["item_id"]): text
            for row, text in generate_rows(
                model, tokenizer, rows, device="cuda", batch_size=batch_size
            )
        }
        timings[batch_size] = time.perf_counter() - started
    changed = [
        item_id for item_id in reads[8] if reads[8][item_id] != reads[16][item_id]
    ]
    result = {
        "kind": "paper2_tm0_7b_score_batch_identity_v1",
        "rows": len(rows),
        "batch_8_seconds": timings[8],
        "batch_16_seconds": timings[16],
        "batch_16_speedup": timings[8] / timings[16],
        "generated_text_exact": not changed,
        "changed_rows": len(changed),
        "changed_item_ids": changed,
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    path = Path("/content/tm0_dry/teacher_7b_batch_identity.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
