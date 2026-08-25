"""Measure the TM-0 7B correctness reader by battery without scoring seals."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/content/tm0_repo")
sys.path.insert(0, str(ROOT))

from eval.eval_paper2_phase3_p31_references import (  # noqa: E402
    MODEL_SPECS,
    generate_rows,
    score_generated,
    score_mcq_rows,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    output = Path("/content/tm0_dry")
    output.mkdir(parents=True, exist_ok=True)
    probe = read_jsonl(Path("/content/tm0_score_cost_probe.jsonl"))
    panel = read_jsonl(Path("/content/tm0_panel.jsonl"))
    panel_counts = Counter(str(row["battery"]) for row in panel)
    spec = MODEL_SPECS["teacher_7b"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], cache_dir="/content/model-cache"
    )
    tokenizer.padding_side = "right"
    loaded_at = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        cache_dir="/content/model-cache",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    load_seconds = time.perf_counter() - loaded_at
    timings = {}
    for battery in sorted({str(row["battery"]) for row in probe}):
        rows = [row for row in probe if row["battery"] == battery]
        started = time.perf_counter()
        if battery in {"arc_easy", "arc_challenge", "mmlu"}:
            scored = score_mcq_rows(
                model,
                tokenizer,
                rows,
                device="cuda",
                candidate_batch_size=32,
            )
            correct = sum(bool(row["correct"]) for row in scored)
        else:
            generated = list(
                generate_rows(model, tokenizer, rows, device="cuda", batch_size=8)
            )
            correct = sum(score_generated(row, text)[0] for row, text in generated)
        elapsed = time.perf_counter() - started
        timings[battery] = {
            "probe_rows": len(rows),
            "correct_rows": int(correct),
            "seconds": elapsed,
            "seconds_per_row": elapsed / len(rows),
            "panel_rows": panel_counts[battery],
            "projected_panel_seconds": elapsed / len(rows) * panel_counts[battery],
        }
        print(f"tm0_score_dry battery={battery} receipt={timings[battery]}", flush=True)
    result = {
        "kind": "paper2_tm0_7b_correctness_cost_probe_v1",
        "load_seconds": load_seconds,
        "battery_timings": timings,
        "projected_full_score_seconds": sum(
            row["projected_panel_seconds"] for row in timings.values()
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    (output / "teacher_7b_score_dry_run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
