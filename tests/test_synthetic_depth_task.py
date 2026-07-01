from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.synthetic_depth_task import (
    SyntheticDepthConfig,
    apply_mapping,
    build_dataset,
    build_instance,
    build_mcq_sft_row,
    build_mcq_row,
    build_sft_row,
    render_mcq_completion,
    render_mcq_prompt,
    write_synthetic_depth_dataset,
)


def test_instance_has_requested_orbit_depth() -> None:
    instance = build_instance(
        instance_id="depth7",
        n_symbols=16,
        depth=7,
        seed=123,
        num_choices=4,
    )

    visited = [instance.start]
    current = instance.start
    for _ in range(instance.depth):
        current = instance.mapping[current]
        visited.append(current)

    assert len(set(visited)) == instance.depth + 1
    assert current == instance.target
    assert apply_mapping(instance.mapping, instance.start, instance.depth) == instance.target


def test_instance_rejects_depth_too_large_for_distinct_orbit() -> None:
    with pytest.raises(ValueError, match="depth must be < n_symbols"):
        build_instance(
            instance_id="bad",
            n_symbols=4,
            depth=4,
            seed=1,
        )


def test_mcq_row_contains_target_once_and_preserves_metadata() -> None:
    instance = build_instance(
        instance_id="depth3",
        n_symbols=12,
        depth=3,
        seed=9,
        num_choices=4,
    )
    row = build_mcq_row(instance)

    assert row["id"] == "depth3"
    assert row["depth"] == 3
    assert row["target"] == str(instance.target)
    assert list(row["choices"].values()).count(str(instance.target)) == 1
    assert row["choices"][row["answer"]] == str(instance.target)
    assert "Apply f exactly 3 times" in row["question"]


def test_sft_row_uses_depth_as_loop_target_when_not_overridden() -> None:
    instance = build_instance(
        instance_id="depth5",
        n_symbols=12,
        depth=5,
        seed=10,
    )
    row = build_sft_row(instance, max_target_loops=4)

    assert row["completion"].strip() == str(instance.target)
    assert row["target_loop_count"] == 4
    assert row["synthetic_depth"] == 5
    assert "Answer with only the final value" in row["prompt"]


def test_mcq_sft_row_matches_eval_prompt_and_option_text_completion() -> None:
    instance = build_instance(
        instance_id="depth1",
        n_symbols=8,
        depth=1,
        seed=18,
        num_choices=4,
    )

    row = build_mcq_sft_row(instance, max_target_loops=4, score_target="option_text")

    assert row["prompt"] == render_mcq_prompt(instance)
    assert row["completion"] == f" {instance.target}"
    assert row["target_loop_count"] == 1
    assert row["score_target"] == "option_text"
    assert "Answer:" in row["prompt"]
    assert "A. " in row["prompt"]
    assert row["choices"][row["answer"]] == str(instance.target)


def test_mcq_sft_completion_modes() -> None:
    instance = build_instance(
        instance_id="depth2",
        n_symbols=8,
        depth=2,
        seed=19,
        num_choices=4,
    )
    label = "ABCDEF"[instance.answer_index]

    assert render_mcq_completion(instance, score_target="label") == f" {label}"
    assert render_mcq_completion(instance, score_target="option_text") == f" {instance.target}"
    assert render_mcq_completion(instance, score_target="label_and_text") == f" {label}. {instance.target}"
    with pytest.raises(ValueError, match="Unknown score_target"):
        render_mcq_completion(instance, score_target="bad")


def test_dataset_generation_is_deterministic_and_balanced() -> None:
    config = SyntheticDepthConfig(
        n_symbols=10,
        max_depth=4,
        rows_per_depth=3,
        seed=77,
        max_target_loops=4,
    )

    first = build_dataset(config, split="train")
    second = build_dataset(config, split="train")

    assert [row.instance_id for row in first] == [row.instance_id for row in second]
    assert [row.target for row in first] == [row.target for row in second]
    assert {depth: sum(1 for row in first if row.depth == depth) for depth in range(1, 5)} == {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
    }


def test_write_synthetic_depth_dataset_writes_train_val_test_and_summary(tmp_path: Path) -> None:
    summary = write_synthetic_depth_dataset(
        output_dir=tmp_path,
        config=SyntheticDepthConfig(
            n_symbols=9,
            max_depth=3,
            rows_per_depth=2,
            seed=5,
            max_target_loops=3,
        ),
    )

    assert summary["rows"]["train"] == 6
    assert summary["rows"]["val"] == 6
    assert summary["rows"]["test"] == 6
    assert summary["orbit_guarantee"] == "distinct_prefix_length_depth_plus_one"

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_rows = [
        json.loads(line)
        for line in (tmp_path / "test_mcq.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train_rows) == 6
    assert len(test_rows) == 6
    assert all("target_loop_count" in row for row in train_rows)
    assert all("choices" in row and "answer" in row for row in test_rows)
    assert (tmp_path / "train_mcq_option_text_sft.jsonl").exists()
    assert (tmp_path / "train_mcq_label_sft.jsonl").exists()
    assert (tmp_path / "train_mcq_label_and_text_sft.jsonl").exists()
    assert summary["files"]["train"]["mcq_option_text_sft"] == "train_mcq_option_text_sft.jsonl"
