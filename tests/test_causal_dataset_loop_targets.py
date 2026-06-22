from __future__ import annotations

import json

from training.dataset import JsonlCausalDataset


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
