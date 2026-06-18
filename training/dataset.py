"""Small JSONL dataset utilities for early recurrent-depth experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from models.halting import target_loop_counts


class JsonlCausalDataset(Dataset):
    """Reads JSONL rows with either ``text`` or ``prompt`` + ``completion``.

    Optional fields:
        cot: String chain-of-thought/reference reasoning trace for loop target.
        cot_tokens: Precomputed integer CoT token count.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int,
        max_train_loops: int,
        train_on_prompt: bool = False,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_train_loops = max_train_loops
        self.train_on_prompt = train_on_prompt
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = row.get("prompt", "")
        completion = row.get("completion")
        if completion is None:
            text = row["text"]
            prompt_len = 0
        else:
            text = prompt + completion
            prompt_len = len(
                self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
            )

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        if completion is not None and not self.train_on_prompt:
            labels[: min(prompt_len, labels.numel())] = -100

        if "cot_tokens" in row:
            cot_len = int(row["cot_tokens"])
        elif "cot" in row:
            cot_len = len(self.tokenizer(row["cot"], add_special_tokens=False)["input_ids"])
        else:
            cot_len = max(1, labels.ne(-100).sum().item())

        input_len = int(input_ids.numel())
        target_loops = target_loop_counts(input_len, cot_len, self.max_train_loops)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "target_loop_counts": target_loops,
        }


def collate_causal_batch(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(example["input_ids"].numel() for example in examples)
    input_ids = []
    labels = []
    attention_mask = []
    target_counts = []
    for example in examples:
        length = example["input_ids"].numel()
        pad_len = max_len - length
        input_ids.append(torch.nn.functional.pad(example["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(torch.nn.functional.pad(example["labels"], (0, pad_len), value=-100))
        attention_mask.append(torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, pad_len), value=0))
        target_counts.append(example["target_loop_counts"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "target_loop_counts": torch.stack(target_counts).view(-1),
    }
