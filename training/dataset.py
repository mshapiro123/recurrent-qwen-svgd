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
        target_loop_count: Explicit 1-indexed loop-depth target. This is used
            for typed direct/deep curriculum rows and overrides the token-count
            heuristic.
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
        if "target_loop_count" in row:
            target_loops = torch.tensor(
                int(row["target_loop_count"]),
                dtype=torch.long,
            ).clamp(1, self.max_train_loops)
        else:
            target_loops = target_loop_counts(input_len, cot_len, self.max_train_loops)
        item = {
            "input_ids": input_ids,
            "labels": labels,
            "target_loop_counts": target_loops,
        }
        if "loop_completions" in row:
            loop_labels = torch.full(
                (self.max_train_loops, input_ids.numel()),
                -100,
                dtype=torch.long,
            )
            completions = list(row.get("loop_completions") or [])
            for loop_idx, completion_text in enumerate(completions[: self.max_train_loops]):
                if completion_text is None:
                    continue
                loop_text = prompt + str(completion_text)
                loop_encoded = self.tokenizer(
                    loop_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
                if len(loop_encoded) != input_ids.numel():
                    raise ValueError(
                        "loop_completions must tokenize to the same length as prompt + completion "
                        f"for row {row.get('id') or row.get('instance_id') or '<unknown>'}. "
                        f"Got {len(loop_encoded)} vs {input_ids.numel()}."
                    )
                loop_tensor = torch.tensor(loop_encoded, dtype=torch.long)
                loop_labels[loop_idx, : min(prompt_len, loop_labels.shape[1])] = -100
                loop_labels[loop_idx, prompt_len:] = loop_tensor[prompt_len:]
            item["loop_labels"] = loop_labels
        return item


def collate_causal_batch(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(example["input_ids"].numel() for example in examples)
    input_ids = []
    labels = []
    attention_mask = []
    target_counts = []
    has_loop_labels = any("loop_labels" in example for example in examples)
    loop_labels = []
    for example in examples:
        length = example["input_ids"].numel()
        pad_len = max_len - length
        input_ids.append(torch.nn.functional.pad(example["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(torch.nn.functional.pad(example["labels"], (0, pad_len), value=-100))
        attention_mask.append(torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, pad_len), value=0))
        target_counts.append(example["target_loop_counts"])
        if has_loop_labels:
            if "loop_labels" not in example:
                raise ValueError("Cannot mix examples with and without loop_labels in one batch")
            loop_labels.append(torch.nn.functional.pad(example["loop_labels"], (0, pad_len), value=-100))

    batch = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "target_loop_counts": torch.stack(target_counts).view(-1),
    }
    if has_loop_labels:
        batch["loop_labels"] = torch.stack(loop_labels)
    return batch
