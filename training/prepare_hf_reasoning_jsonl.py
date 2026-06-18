"""Convert Hugging Face reasoning datasets into this project's JSONL format."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def extract_thinking(text: str) -> str:
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def split_qwen_text(text: str) -> tuple[str, str] | None:
    marker = "<|im_start|>assistant\n"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    prompt = text[: idx + len(marker)]
    completion = text[idx + len(marker) :]
    return prompt, completion


def messages_to_prompt_completion(row: dict[str, Any], tokenizer: Any) -> tuple[str, str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        return None

    prompt_messages = messages[:last_assistant_idx]
    assistant_content = messages[last_assistant_idx].get("content", "")
    if not assistant_content:
        return None

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = assistant_content
    return prompt, completion, extract_thinking(completion)


def raw_thinking_response_to_prompt_completion(
    row: dict[str, Any],
    tokenizer: Any,
) -> tuple[str, str, str] | None:
    thinking = row.get("thinking")
    response = row.get("response") or row.get("solution") or row.get("answer")
    if not thinking or not response:
        return None

    if isinstance(row.get("messages"), list):
        prompt_messages = [msg for msg in row["messages"] if msg.get("role") != "assistant"]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        problem = row.get("problem") or row.get("question") or row.get("prompt") or row.get("instruction")
        if not problem:
            return None
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(problem)}],
            tokenize=False,
            add_generation_prompt=True,
        )

    completion = f"<think>\n{thinking.strip()}\n</think>\n\n{str(response).strip()}"
    return prompt, completion, str(thinking)


def text_to_prompt_completion(row: dict[str, Any]) -> tuple[str, str, str] | None:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    split = split_qwen_text(text)
    if split is None:
        return "", text, extract_thinking(text)
    prompt, completion = split
    return prompt, completion, extract_thinking(completion)


def row_to_example(row: dict[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    converted = (
        raw_thinking_response_to_prompt_completion(row, tokenizer)
        or messages_to_prompt_completion(row, tokenizer)
        or text_to_prompt_completion(row)
    )
    if converted is None:
        return None

    prompt, completion, thinking = converted
    if not completion.strip():
        return None

    cot_tokens = len(tokenizer(thinking, add_special_tokens=False)["input_ids"]) if thinking else 1
    return {
        "prompt": prompt,
        "completion": completion,
        "cot_tokens": max(1, cot_tokens),
        "source_dataset": row.get("source_dataset"),
        "category": row.get("category") or row.get("domain"),
        "difficulty": row.get("difficulty"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--val_jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_total_tokens", type=int, default=4096)
    parser.add_argument("--min_completion_tokens", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    ds = load_dataset(args.dataset_id, split=args.split)
    rows = list(ds)
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    examples = []
    skipped = 0
    for row in rows:
        example = row_to_example(row, tokenizer)
        if example is None:
            skipped += 1
            continue

        prompt_tokens = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
        completion_tokens = len(tokenizer(example["completion"], add_special_tokens=False)["input_ids"])
        if completion_tokens < args.min_completion_tokens:
            skipped += 1
            continue
        if prompt_tokens + completion_tokens > args.max_total_tokens:
            skipped += 1
            continue
        examples.append(example)

    if args.val_jsonl:
        val_count = max(1, int(len(examples) * args.val_fraction))
        val_examples = examples[:val_count]
        train_examples = examples[val_count:]
    else:
        train_examples = examples
        val_examples = []

    write_jsonl(args.output_jsonl, train_examples)
    if args.val_jsonl:
        write_jsonl(args.val_jsonl, val_examples)

    print(f"dataset_id={args.dataset_id}")
    print(f"train_rows={len(train_examples)}")
    print(f"val_rows={len(val_examples)}")
    print(f"skipped_rows={skipped}")
    return 0


def write_jsonl(path: str | Path, examples: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
