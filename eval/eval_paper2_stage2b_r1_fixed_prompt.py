"""R-1 fixed-prompt first-token probe for cross-runtime attribution."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
)
from eval.eval_paper2_phase3_p34_task_inference import P34TaskInferenceGraph
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition, sha256_file
from eval.eval_paper2_stage2a_cv1 import _load_host_memory


REGISTERED_ITEM_ID = "gsm8k-evaluation-1156"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--t3a_checkpoint", type=Path, required=True)
    parser.add_argument("--memory_slots", type=int, required=True)
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
    parser.add_argument("--runtime_label", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    panel = {str(row["item_id"]): row for row in read_jsonl(args.panel)}
    row = panel.get(REGISTERED_ITEM_ID)
    if row is None or row.get("battery") != "gsm8k" or row.get("partition") != "dev":
        raise RuntimeError("registered R-1 fixed prompt is absent from the frozen DEV panel")
    checkpoint = torch.load(args.t3a_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != "t3a" or int(checkpoint.get("seed", -1)) != 0:
        raise RuntimeError("R-1 requires the seed-0 T3a endpoint")
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    model_spec = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model"], revision=model_spec["revision"], cache_dir=args.model_cache
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["model"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    sidecar, chain = load_condition(
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
    sidecar.to("cuda").eval()
    memory = _load_host_memory(
        host="t3a",
        checkpoint=checkpoint,
        geometry=geometry,
        memory_slots=args.memory_slots,
    ).to("cuda").eval()
    graph = P34TaskInferenceGraph(
        base_model=model,
        sidecar=sidecar,
        flow_loops=4,
        stage2a_memory_system=memory,
        stage2a_geometry=geometry,
        stage2a_amplitude=0.05,
        stage2a_value_scale=1.0,
    )
    content, _cap = _generation_prompt(row)
    prompt = _chat_prompt(tokenizer, content)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    with torch.inference_mode():
        output = graph.next_token(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )
    logits = output.augmented_logits[0].float().cpu().contiguous()
    top = logits.topk(10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = args.output_dir / "first_token_logits.pt"
    torch.save(logits, logits_path)
    manifest = {
        "kind": "paper2_stage2b_r1_fixed_prompt_v1",
        "status": "complete_score_only",
        "runtime_label": args.runtime_label,
        "item_id": REGISTERED_ITEM_ID,
        "panel_sha256": sha256_file(args.panel),
        "t3a_checkpoint_sha256": sha256_file(args.t3a_checkpoint),
        "p35_checkpoint_sha256": args.p35_sha256,
        "first_token_logits_sha256": sha256_file(logits_path),
        "top_token_ids": [int(value) for value in top.indices.tolist()],
        "top_logits": [float(value) for value in top.values.tolist()],
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "attention_backend": "sdpa",
            "weights_dtype": "bfloat16",
        },
        "checkpoint_chain": chain,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
