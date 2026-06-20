from __future__ import annotations

from eval.arc_agi_utils import ArcAgiExample, ArcPair
from training.prepare_arc_agi_sft_jsonl import example_to_jsonl_row


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": text.split()}

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        del tokenize, add_generation_prompt
        return messages[0]["content"] + "\nassistant:\n"


def test_example_to_jsonl_row_can_include_symbolic_trace() -> None:
    example = ArcAgiExample(
        task_id="constant",
        test_index=0,
        train=(
            ArcPair(input=[[1]], output=[[9, 9]]),
            ArcPair(input=[[2]], output=[[9, 9]]),
        ),
        test_input=[[3]],
        test_output=[[9, 9]],
    )
    row = example_to_jsonl_row(
        FakeTokenizer(),
        example,
        append_eos=True,
        source="unit",
        output_format="compact",
        trace_mode="symbolic",
    )
    assert row is not None
    assert "You may include a brief <think> trace" in str(row["prompt"])
    assert str(row["completion"]).startswith("<think>")
    assert "99" in str(row["completion"])
    assert row["trace_mode"] == "symbolic"
    assert row["trace_source"] == "constant_output"
    assert int(row["cot_tokens"]) > 1
