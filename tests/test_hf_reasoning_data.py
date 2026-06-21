from __future__ import annotations

from training.inspect_hf_reasoning_dataset import audit_rows
from training.prepare_hf_reasoning_jsonl import row_to_example, row_to_prompt_completion


class DummyTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        tokens = str(text).split()
        if max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}


def test_fable_flat_converter_builds_think_completion() -> None:
    row = {
        "context": "Build a tiny solver.",
        "cot": "Plan the file change, then test it.",
        "output_type": "assistant_text",
        "output": "Implemented and tested.",
    }

    example = row_to_example(
        row,
        DummyTokenizer(),
        adapter="fable_flat",
        source_dataset_name="Glint-Research/Fable-5-traces",
    )

    assert example is not None
    assert "Build a tiny solver" in example["prompt"]
    assert str(example["completion"]).startswith("<think>\nPlan the file change")
    assert "Implemented and tested." in str(example["completion"])
    assert example["source_dataset"] == "Glint-Research/Fable-5-traces"
    assert example["category"] == "assistant_text"
    assert int(example["cot_tokens"]) > 1


def test_fable_pi_agent_converter_rejects_tool_only_rows() -> None:
    row = {
        "harness": "pi",
        "trace": "{\"type\":\"tool_call\"}",
        "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "Create a game."}],
    }

    assert row_to_prompt_completion(row, DummyTokenizer(), adapter="fable_pi_agent") is None


def test_fable_pi_agent_converter_accepts_assistant_text_content_list() -> None:
    row = {
        "harness": "pi",
        "trace": "{\"type\":\"thinking\"}",
        "messages": [
            {"role": "user", "content": "Refactor this."},
            {"role": "assistant", "content": [{"type": "text", "text": "<think>\nCheck call sites.\n</think>\nDone."}]},
        ],
    }

    converted = row_to_prompt_completion(row, DummyTokenizer(), adapter="fable_pi_agent")

    assert converted is not None
    prompt, completion, thinking = converted
    assert "Refactor this" in prompt
    assert "Done." in completion
    assert thinking == "Check call sites."


def test_audit_rows_marks_fable_as_later_agent_trace_source() -> None:
    rows = [
        {
            "context": "Build a tiny solver.",
            "cot": "Think through the plan.",
            "output": "Done.",
            "tools": [{"type": "function"}],
            "num_tool_calls": 1,
        }
    ]

    report = audit_rows(
        rows,
        DummyTokenizer(),
        dataset_id="Glint-Research/Fable-5-traces",
        adapter="auto",
    )

    assert report["converted_rows"] == 1
    assert report["adapter_success_counts"]["fable_flat"] == 1
    assert report["training_role"]["priority"] == "later"
    assert report["training_role"]["primary_role"] == "agent_tool_trace_or_coding_diversity"


def test_audit_rows_marks_qwen_text_opus_as_immediate_candidate() -> None:
    rows = [
        {
            "text": (
                "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                "<|im_start|>user\nSolve 2+2.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\nAdd numbers.\n</think>\n4<|im_end|>"
            )
        }
    ]

    report = audit_rows(
        rows,
        DummyTokenizer(),
        dataset_id="lordx64/reasoning-distill-opus-4-7-max-sft",
        adapter="auto",
    )

    assert report["converted_rows"] == 1
    assert report["adapter_success_counts"]["qwen_text"] == 1
    assert report["training_role"]["priority"] == "immediate_candidate"
