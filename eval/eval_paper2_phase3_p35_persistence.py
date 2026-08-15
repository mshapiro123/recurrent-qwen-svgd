"""No-training P3.5 fresh-versus-carried scratch persistence probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.cache_paper2_phase3_agreement_oracle import analytic_oracle_directions
from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
    score_generated,
)
from eval.eval_paper2_phase3_p34_task_inference import P34TaskInferenceGraph
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition, read_jsonl, write_json
from training.paper2_phase3_p35 import P35_PRIMARY_EVAL_CEILING
from training.paper2_phase3_p33_prep import sha256_file


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


@torch.inference_mode()
def _generate_batch(
    *,
    graph: P34TaskInferenceGraph,
    tokenizer: Any,
    prompts: list[str],
    caps: list[int],
    device: str,
) -> tuple[list[list[int]], list[list[int]]]:
    encoded = tokenizer(
        [_chat_prompt(tokenizer, prompt) for prompt in prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)
    state, output = graph.prefill_cached(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    )
    generated = [[] for _prompt in prompts]
    source_tokens = [[] for _prompt in prompts]
    active = [True for _prompt in prompts]
    for _token_index in range(max(caps)):
        selected = output.augmented_logits.argmax(dim=-1)
        source = output.base_logits.argmax(dim=-1)
        for index, (token, source_token) in enumerate(
            zip(selected.tolist(), source.tolist())
        ):
            if not active[index]:
                continue
            generated[index].append(int(token))
            source_tokens[index].append(int(source_token))
            if (
                len(generated[index]) >= caps[index]
                or tokenizer.eos_token_id is not None
                and int(token) == int(tokenizer.eos_token_id)
            ):
                active[index] = False
        if not any(active):
            break
        state, output = graph.advance_cached(
            state=state, selected_tokens=selected
        )
    return generated, source_tokens


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("P3.5 persistence batch size must be positive")
    population = [
        row for row in read_jsonl(args.panel)
        if str(row["battery"]) in {"gsm8k", "mbpp"}
    ]
    if not population or any(str(row["partition"]) != "dev" for row in population):
        raise RuntimeError("P3.5 persistence probe requires DEV GSM8K and MBPP rows")
    panel = []
    for battery in ("gsm8k", "mbpp"):
        selected = [row for row in population if str(row["battery"]) == battery]
        selected.sort(
            key=lambda row: hashlib.sha256(
                f"20260815:{battery}:{row['document_id']}:{row['item_id']}".encode()
            ).hexdigest()
        )
        panel.extend(selected[:128])
    if any(token in str(args.panel).casefold() for token in ("confirm", "eval_e")):
        raise RuntimeError("P3.5 persistence probe cannot contact sealed data")
    spec = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
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
    )
    sidecar.bridge.set_gate_ceiling(P35_PRIMARY_EVAL_CEILING)
    fresh = P34TaskInferenceGraph(
        base_model=model, sidecar=sidecar, cross_token_persistence=False
    )
    carried = P34TaskInferenceGraph(
        base_model=model, sidecar=sidecar, cross_token_persistence=True
    )
    lm_head = model.get_output_embeddings().weight.detach()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(panel), args.batch_size):
        batch = panel[start : start + args.batch_size]
        prompt_caps = [_generation_prompt(row) for row in batch]
        prompts = [value[0] for value in prompt_caps]
        caps = [value[1] for value in prompt_caps]
        fresh_tokens, _fresh_sources = _generate_batch(
            graph=fresh,
            tokenizer=tokenizer,
            prompts=prompts,
            caps=caps,
            device=args.device,
        )
        carry_tokens, carry_sources = _generate_batch(
            graph=carried,
            tokenizer=tokenizer,
            prompts=prompts,
            caps=caps,
            device=args.device,
        )
        for row, fresh_row, carry_row, sources_row in zip(
            batch, fresh_tokens, carry_tokens, carry_sources
        ):
            pairs = [
                (source_token, target_token)
                for source_token, target_token in zip(sources_row, fresh_row)
                if source_token != target_token
            ]
            if pairs:
                direction = analytic_oracle_directions(
                    lm_head_weight=lm_head,
                    source_tokens=torch.tensor(
                        [pair[0] for pair in pairs], device=args.device
                    ),
                    target_tokens=torch.tensor(
                        [pair[1] for pair in pairs], device=args.device
                    ),
                )
                if not bool(torch.isfinite(direction).all()):
                    raise RuntimeError(
                        "P3.5 persistence re-anchor produced non-finite direction"
                    )
            fresh_text = tokenizer.decode(fresh_row, skip_special_tokens=True)
            carry_text = tokenizer.decode(carry_row, skip_special_tokens=True)
            fresh_correct, fresh_prediction = score_generated(row, fresh_text)
            carry_correct, carry_prediction = score_generated(row, carry_text)
            difference = _first_difference(fresh_row, carry_row)
            rows.append(
                {
                    "item_id": str(row["item_id"]),
                    "document_id": str(row["document_id"]),
                    "battery": str(row["battery"]),
                    "fresh_correct": bool(fresh_correct),
                    "carried_correct": bool(carry_correct),
                    "fresh_prediction": fresh_prediction,
                    "carried_prediction": carry_prediction,
                    "fresh_tokens": fresh_row,
                    "carried_tokens": carry_row,
                    "first_token_difference": difference,
                    "later_token_changed": difference not in (None, 0),
                    "on_the_fly_reanchors": len(pairs),
                }
            )
        print(f"p35_persistence_progress rows={len(rows)}/{len(panel)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    by_battery = {}
    for battery in ("gsm8k", "mbpp"):
        selected = [row for row in rows if row["battery"] == battery]
        by_battery[battery] = {
            "rows": len(selected),
            "fresh_correct": sum(row["fresh_correct"] for row in selected),
            "carried_correct": sum(row["carried_correct"] for row in selected),
            "net_carried_minus_fresh": sum(
                int(row["carried_correct"]) - int(row["fresh_correct"])
                for row in selected
            ),
            "later_token_changed_rows": sum(row["later_token_changed"] for row in selected),
        }
    summary = {
        "kind": "paper2_phase3_p35_persistence_probe_v1",
        "status": "complete_dev_only_no_training",
        "rows": len(rows),
        "selection_seed": 20260815,
        "selection_rule": "SHA256(seed:battery:document_id:item_id), first 128 per battery or all if fewer",
        "panel_sha256": sha256_file(args.panel),
        "rows_sha256": sha256_file(rows_path),
        "checkpoint_chain": chain,
        "gate_ceiling": P35_PRIMARY_EVAL_CEILING,
        "fresh_scratch_per_token": True,
        "controlled_cross_token_carry": True,
        "oracle_cache_used": False,
        "direction_telemetry": {
            "reanchored_from_current_source_every_token": True,
            "target": "matched fresh-path selected token",
            "nontrivial_reanchors": sum(row["on_the_fly_reanchors"] for row in rows),
            "frozen_source_direction_reuse": False,
        },
        "by_battery": by_battery,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del fresh, carried, sidecar, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    for name in ("migrated", "p33", "i1", "p34"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
