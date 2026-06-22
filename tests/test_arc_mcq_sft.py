from __future__ import annotations

from training.prepare_arc_mcq_sft_jsonl import arc_row_to_sft


def test_arc_row_to_sft_matches_eval_mcq_label_surface() -> None:
    row = {
        "id": "arc-1",
        "question": "What is 2 + 2?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["3", "4", "5", "6"]},
        "answerKey": "B",
        "config": "ARC-Easy",
    }

    example = arc_row_to_sft(
        row,
        index=0,
        seed=0,
        shuffle_choices=False,
        prompt_style="with_options",
        score_target="label",
    )

    assert example["prompt"] == "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
    assert example["completion"] == " B"
    assert example["cot_tokens"] == 1
    assert example["source_dataset"] == "ai2_arc"


def test_arc_row_to_sft_can_emit_routing_loop_target() -> None:
    row = {
        "id": "arc-1",
        "question": "What is 2 + 2?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["3", "4", "5", "6"]},
        "answerKey": "B",
        "config": "ARC-Easy",
    }

    example = arc_row_to_sft(
        row,
        index=0,
        seed=0,
        shuffle_choices=False,
        prompt_style="with_options",
        score_target="label",
        target_loop_count=1,
        routing_type="direct",
    )

    assert example["target_loop_count"] == 1
    assert example["routing_type"] == "direct"
