"""Simple multiple-choice scorer for GPQA-style JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split
from models.recurrent_wrapper import RecurrentQwenForCausalLM


def option_items(row: dict) -> list[tuple[str, str]]:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, dict):
        return list(choices.items())
    if isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E", "F"]
        return list(zip(labels, choices))
    raise ValueError("Each row must contain choices/options as a list or dict")


def format_prompt(row: dict, options: list[tuple[str, str]]) -> str:
    question = row.get("question") or row.get("prompt")
    if question is None:
        raise ValueError("Each row must contain question or prompt")
    rendered = "\n".join(f"{label}. {text}" for label, text in options)
    return f"{question}\n{rendered}\nAnswer:"


def score_completion(wrapper, tokenizer, prompt: str, completion: str, device: str, max_loops: int) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=True).to(device)
    labels = full["input_ids"].clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    with torch.no_grad():
        output = wrapper(
            input_ids=full["input_ids"],
            attention_mask=full["attention_mask"],
            labels=labels,
            max_loops=max_loops,
            use_cache=False,
            return_dict=True,
        )
    scored_tokens = labels.ne(-100).sum().item()
    return -float(output.metrics["expected_ce"]) * max(scored_tokens, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    model.eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    wrapper.eval()

    correct = 0
    total = 0
    for line in Path(args.data_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        options = option_items(row)
        prompt = format_prompt(row, options)
        scores = {
            label: score_completion(wrapper, tokenizer, prompt, f" {label}", args.device, args.max_loops)
            for label, _ in options
        }
        prediction = max(scores, key=scores.get)
        answer = str(row.get("answer") or row.get("label") or row.get("target")).strip()
        total += 1
        correct += int(prediction == answer)
        print(json.dumps({"prediction": prediction, "answer": answer, "scores": scores}))

    accuracy = correct / max(total, 1)
    print(f"accuracy={accuracy:.4f} total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
