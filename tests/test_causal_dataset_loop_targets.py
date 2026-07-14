from __future__ import annotations

import json

import torch

from training.dataset import JsonlCausalDataset
from training.dataset import collate_causal_batch


class TinyTokenizer:
    eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        tokens = str(text).split()
        if add_special_tokens:
            tokens = ["<bos>", *tokens]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}


def test_jsonl_dataset_honors_explicit_target_loop_count(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Question:",
                "completion": " A",
                "cot_tokens": 1,
                "target_loop_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = JsonlCausalDataset(
        path,
        tokenizer=TinyTokenizer(),
        max_length=32,
        max_train_loops=4,
        train_on_prompt=False,
    )

    assert int(dataset[0]["target_loop_counts"]) == 3


def test_jsonl_dataset_clamps_explicit_target_loop_count(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Question:",
                "completion": " A",
                "cot_tokens": 1,
                "target_loop_count": 99,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = JsonlCausalDataset(
        path,
        tokenizer=TinyTokenizer(),
        max_length=32,
        max_train_loops=4,
        train_on_prompt=False,
    )

    assert int(dataset[0]["target_loop_counts"]) == 4


def test_jsonl_dataset_builds_and_collates_loop_labels(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Question:",
                "completion": " A",
                "loop_completions": [" A", " B"],
                "target_loop_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = TinyTokenizer()
    dataset = JsonlCausalDataset(
        path,
        tokenizer=tokenizer,
        max_length=32,
        max_train_loops=4,
        train_on_prompt=False,
    )

    item = dataset[0]
    assert item["loop_labels"].shape == (4, item["input_ids"].numel())
    active_count = item["labels"].ne(-100).sum()
    assert active_count == 1
    assert item["loop_labels"][0].ne(-100).sum() == active_count
    assert item["loop_labels"][1].ne(-100).sum() == active_count
    assert item["loop_labels"][2].eq(-100).all()

    batch = collate_causal_batch([item], pad_token_id=0)
    assert batch["loop_labels"].shape == (1, 4, item["input_ids"].numel())
    assert torch.equal(batch["target_loop_counts"], torch.tensor([2]))


def test_jsonl_dataset_collates_optional_row_specific_loop_weights(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "prompt": "Question:",
            "completion": " A",
            "loop_completions": [" A", " B"],
            "target_loop_count": 2,
            "loop_label_weights": [0.5, 1.5, 0.0],
            "forward_loop_count": 2,
        },
        {
            "prompt": "Question:",
            "completion": " B",
            "loop_completions": [" B", " C", " D"],
            "target_loop_count": 3,
            "loop_label_weights": [0.25, 0.25, 2.5],
            "forward_loop_count": 3,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = JsonlCausalDataset(
        path,
        tokenizer=TinyTokenizer(),
        max_length=32,
        max_train_loops=3,
        train_on_prompt=False,
    )

    batch = collate_causal_batch([dataset[0], dataset[1]], pad_token_id=0)

    assert torch.equal(
        batch["loop_label_weights"],
        torch.tensor([[0.5, 1.5, 0.0], [0.25, 0.25, 2.5]], dtype=torch.float32),
    )
    assert torch.equal(batch["forward_loop_counts"], torch.tensor([2, 3]))


def test_jsonl_dataset_renders_question_rows_like_active_label_eval(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Who has the key?",
                "prompt_style": "question_only",
                "completion": " Kim",
                "loop_completions": [" Jon", " Kim"],
                "target_loop_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class LoggingTokenizer(TinyTokenizer):
        def __init__(self) -> None:
            self.seen: list[tuple[str, bool]] = []

        def __call__(
            self,
            text: str,
            add_special_tokens: bool = True,
            truncation: bool = False,
            max_length: int | None = None,
        ) -> dict[str, list[int]]:
            self.seen.append((text, add_special_tokens))
            return super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )

    tokenizer = LoggingTokenizer()
    dataset = JsonlCausalDataset(
        path,
        tokenizer=tokenizer,
        max_length=32,
        max_train_loops=4,
        train_on_prompt=False,
    )

    item = dataset[0]

    rendered_prompt = "Who has the key?\nAnswer:"
    assert (rendered_prompt, True) in tokenizer.seen
    assert (rendered_prompt + " Kim", True) in tokenizer.seen
    assert (rendered_prompt + " Jon", True) in tokenizer.seen
    active_count = item["labels"].ne(-100).sum()
    assert active_count == 1
    assert item["loop_labels"][0].ne(-100).sum() == active_count
    assert item["loop_labels"][1].ne(-100).sum() == active_count
