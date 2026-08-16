"""Recover the exploratory MBPP sequence likelihood omitted by a Drive flush failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_kp1_t1 import canonical_prompt
from training.paper2_phase3_kp1r_t1_teacher import answer_token_ids


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--gap_rows", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_score_only" or not lock.get("locked_before_scoring"):
        raise RuntimeError("KP-1R lock is not active")
    panel = read_jsonl(args.panel)
    gap = read_jsonl(args.gap_rows)
    panel_by_id = {str(row["item_id"]): row for row in panel}
    mbpp_gap = [row for row in gap if str(row["battery"]) == "mbpp"]
    if len(mbpp_gap) != 25:
        raise RuntimeError("MBPP exploratory recovery population changed")
    if any(str(panel_by_id[str(row["item_id"])]["partition"]) != "dev" for row in mbpp_gap):
        raise RuntimeError("MBPP recovery may read DEV only")

    spec = lock["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], cache_dir=args.model_cache
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        cache_dir=args.model_cache,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    for index, gap_row in enumerate(mbpp_gap):
        row = panel_by_id[str(gap_row["item_id"])]
        prompt_ids = tokenizer(
            canonical_prompt(row, tokenizer), add_special_tokens=True
        )["input_ids"]
        answer_ids = answer_token_ids(row, tokenizer)
        capped = answer_ids[:128]
        inputs = torch.tensor([prompt_ids + capped[:-1]], dtype=torch.long)
        with torch.inference_mode():
            output = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=False,
                return_dict=True,
            )
        positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(capped))
        targets = torch.tensor(capped, dtype=torch.long)
        logp = torch.log_softmax(output.logits[0, positions].float(), dim=-1).gather(
            1, targets[:, None]
        )[:, 0]
        rows.append(
            {
                "item_id": str(gap_row["item_id"]),
                "tokens": len(capped),
                "truncated_to_128": len(answer_ids) > 128,
                "base_mean_log_probability": float(logp.mean()),
            }
        )
        print(f"kp1r_mbpp_recovery_progress rows={index + 1}/25", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "mbpp_teacher_forced_sequence_log_likelihood.jsonl"
    write_jsonl(rows_path, rows)
    summary = {
        "kind": "paper2_phase3_kp1r_mbpp_recovery_summary_v1",
        "status": "complete_cpu_score_only_recovery",
        "rows": len(rows),
        "mean_native_base_log_probability": float(
            sum(row["base_mean_log_probability"] for row in rows) / len(rows)
        ),
        "truncated_rows": sum(bool(row["truncated_to_128"]) for row in rows),
        "row_receipt": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
        "assertions": {
            "dev_only": True,
            "mbpp_exploratory_only": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
    }
    write_json(args.output_dir / "mbpp_recovery_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
