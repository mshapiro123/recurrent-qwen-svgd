"""Evaluate one Stage 2A memory endpoint on the frozen DEV panel only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_inference import P34TaskInferenceGraph
from eval.eval_paper2_phase3_p34_task_trajectory import (
    load_condition,
    score_generation,
    score_mcq,
)
from models.sidecar_v2 import LiteralNGramMemory
from training.paper2_phase3_p31_completion import sha256_file
from training.paper2_stage2a_lock import assert_stage2a_training_authorized
from training.paper2_stage2a_runtime import Stage2AMemorySystem


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def exact_sign_test(fixes: int, regressions: int) -> float:
    discordant = int(fixes) + int(regressions)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(fixes, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("t3a", "t3b", "shuffled", "random"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--p34", type=Path, required=True)
    parser.add_argument("--p34_sha256", required=True)
    parser.add_argument("--p35", type=Path, required=True)
    parser.add_argument("--p35_sha256", required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mcq_batch_size", type=int, default=32)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    args = parser.parse_args()

    lock = read_json(args.lock)
    assert_stage2a_training_authorized(lock)
    panel = read_jsonl(args.panel)
    if len(panel) != 1_024 or any(str(row.get("partition")) != "dev" for row in panel):
        raise RuntimeError("Stage 2A scorer requires the frozen 1,024-row DEV panel")
    if sha256_file(args.panel) != lock["data_separation"]["panel_manifest_sha256"]:
        raise RuntimeError("Stage 2A DEV panel SHA changed")
    base = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    if set(base) != {str(row["item_id"]) for row in panel}:
        raise RuntimeError("Stage 2A base-score coverage changed")
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    if sha256_file(args.geometry) != lock["geometry_fit"]["artifact_sha256"]:
        raise RuntimeError("Stage 2A geometry SHA changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != args.arm or int(checkpoint.get("seed", -1)) != args.seed:
        raise RuntimeError("Stage 2A endpoint identity changed")
    if int(checkpoint.get("step", -1)) != 1_200:
        raise RuntimeError("Stage 2A primary read requires step 1200")

    spec = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], cache_dir=args.model_cache
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    sidecar, checkpoint_receipts = load_condition(
        embedding_weight=model.get_output_embeddings().weight.detach().cpu(),
        migrated=args.migrated,
        migrated_sha256=args.migrated_sha256,
        p33=args.p33,
        p33_sha256=args.p33_sha256,
        i1=args.i1,
        i1_sha256=args.i1_sha256,
        p34=args.p34,
        p34_sha256=args.p34_sha256,
        p35=args.p35,
        p35_sha256=args.p35_sha256,
        control_reader="mean",
    )
    sidecar.bridge.set_gate_ceiling(0.02)
    memory = Stage2AMemorySystem(
        arm=args.arm,
        memory_slots=int(lock["data_separation"]["memory_slots"]),
        memory_keys=geometry["memory_keys"].float(),
        teacher_values=geometry["teacher_values"].float(),
        seed=args.seed,
    )
    memory.load_state_dict(checkpoint["raw_state"])
    named = dict(memory.named_parameters())
    with torch.no_grad():
        for name, value in checkpoint["ema_state"].items():
            if name not in named:
                raise RuntimeError(f"Stage 2A EMA contains unknown parameter: {name}")
            named[name].copy_(value.to(dtype=named[name].dtype))
    memory.to("cuda").eval()
    graph = P34TaskInferenceGraph(
        base_model=model,
        sidecar=sidecar,
        flow_loops=4,
        stage2a_memory_system=memory,
        stage2a_geometry=geometry,
        stage2a_amplitude=0.05,
    )
    mcq = [row for row in panel if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}]
    generated = [row for row in panel if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    scored = score_mcq(graph, tokenizer, mcq, batch_size=args.mcq_batch_size)
    scored.extend(
        score_generation(
            graph, tokenizer, generated, batch_size=args.generation_batch_size
        )
    )
    by_id = {str(row["item_id"]): row for row in scored}
    if set(by_id) != set(base):
        raise RuntimeError("Stage 2A DEV output coverage changed")
    source = {str(row["item_id"]): row for row in panel}
    rows = []
    for item_id in sorted(by_id):
        result = by_id[item_id]
        rows.append(
            {
                "kind": "paper2_stage2a_dev_row_v1",
                "arm": args.arm,
                "seed": args.seed,
                "partition": "dev",
                "item_id": item_id,
                "battery": source[item_id]["battery"],
                "battery_role": source[item_id]["battery_role"],
                "base_correct": bool(base[item_id]["correct"]),
                **result,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "dev_rows.jsonl"
    write_jsonl(rows_path, rows)

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        fixes = sum(not row["base_correct"] and row["augmented_correct"] for row in selected)
        regressions = sum(row["base_correct"] and not row["augmented_correct"] for row in selected)
        base_correct = sum(row["base_correct"] for row in selected)
        augmented_correct = sum(row["augmented_correct"] for row in selected)
        return {
            "rows": len(selected),
            "base_correct": base_correct,
            "augmented_correct": augmented_correct,
            "delta_rows": augmented_correct - base_correct,
            "fixes": fixes,
            "regressions": regressions,
            "paired_sign_test_p_two_sided": exact_sign_test(fixes, regressions),
            "memory_compatibility_gate_mean": (
                sum(float(row.get("memory_compatibility_gate_mean", 1.0)) for row in selected)
                / len(selected)
            ),
        }

    by_battery: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battery[str(row["battery"])].append(row)
    summary = {
        "kind": "paper2_stage2a_dev_summary_v1",
        "status": "complete_dev_only",
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "endpoint": "ema_primary_step_1200",
        "registered_amplitude": 0.05,
        "pooled": summarize(rows),
        "by_battery": {
            battery: summarize(selected)
            for battery, selected in sorted(by_battery.items())
        },
        "retrieval_slots_observed": len(
            {
                slot
                for row in rows
                for read in row.get("memory_top_k_slot_ids", [])
                for slot in read
            }
        ),
        "dev_rows_sha256": sha256_file(rows_path),
        "checkpoint_receipts": checkpoint_receipts,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
