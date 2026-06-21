from __future__ import annotations

from training.prepare_arc_agi_candidate_distill_jsonl import (
    build_distill_rows,
    choose_exact_candidates,
    completion_from_candidate,
    extract_think_trace,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": text.split()}


def candidate_row(
    idx: int,
    grid: list[list[int]] | None,
    *,
    exact: bool,
    source: str = "model",
    selected: bool = False,
    prompt: str = "prompt\nassistant:\n",
    candidate_text: str | None = None,
    parse_method: str = "grid",
    program_fits_train: bool = False,
) -> dict:
    return {
        "task_id": "task",
        "test_index": 0,
        "candidate_index": idx,
        "candidate_source": source,
        "candidate_text": candidate_text if candidate_text is not None else str(grid),
        "prompt": prompt,
        "parsed_grid": grid,
        "parse_method": parse_method,
        "program_fits_train": program_fits_train,
        "selected": selected,
        "score": {"valid": grid is not None, "exact": exact},
    }


def test_choose_exact_candidates_prefers_selected_exact() -> None:
    rows = [
        candidate_row(0, [[1]], exact=True, source="symbolic_copy"),
        candidate_row(1, [[2]], exact=True, selected=True),
        candidate_row(2, [[3]], exact=False),
    ]

    chosen = choose_exact_candidates(rows, choice="best_exact")

    assert [row["candidate_index"] for row in chosen] == [1]


def test_choose_exact_candidates_prefers_verified_program_when_no_selected_exact() -> None:
    rows = [
        candidate_row(0, [[1]], exact=True, source="symbolic_copy"),
        candidate_row(
            1,
            [[2]],
            exact=True,
            source="model_program",
            parse_method="program",
            program_fits_train=True,
        ),
    ]

    chosen = choose_exact_candidates(rows, choice="best_exact")

    assert [row["candidate_index"] for row in chosen] == [1]


def test_choose_selected_exact_returns_empty_when_selector_missed() -> None:
    rows = [
        candidate_row(0, [[1]], exact=False, selected=True),
        candidate_row(1, [[2]], exact=True),
    ]

    assert choose_exact_candidates(rows, choice="selected_exact") == []


def test_choose_selector_exact_returns_only_selector_generated_exact_rows() -> None:
    rows = [
        candidate_row(0, [[1]], exact=True, selected=True, source="model_tta_identity"),
        {
            **candidate_row(1, [[2]], exact=True, selected=True, source="selector_cell_vote"),
            "selector_generated": True,
        },
        candidate_row(2, [[3]], exact=False, source="selector_cell_vote"),
    ]

    chosen = choose_exact_candidates(rows, choice="selector_exact")

    assert [row["candidate_index"] for row in chosen] == [1]


def test_choose_selector_exact_returns_empty_without_selector_generated_exact() -> None:
    rows = [
        candidate_row(0, [[1]], exact=True, selected=True, source="model_tta_identity"),
        candidate_row(1, [[2]], exact=False, source="selector_cell_vote"),
    ]

    assert choose_exact_candidates(rows, choice="selector_exact") == []


def test_all_exact_returns_ranked_exact_rows() -> None:
    rows = [
        candidate_row(0, [[1]], exact=False),
        candidate_row(1, [[2]], exact=True, source="model", candidate_text="long answer"),
        candidate_row(2, [[3]], exact=True, source="symbolic_copy", candidate_text="short"),
    ]

    chosen = choose_exact_candidates(rows, choice="all_exact")

    assert [row["candidate_index"] for row in chosen] == [2, 1]


def test_completion_from_candidate_can_emit_canonical_grid() -> None:
    row = candidate_row(0, [[1, 2], [3, 4]], exact=True, candidate_text="messy but exact")

    assert completion_from_candidate(row, completion_source="canonical_grid", output_format="compact") == "12\n34"


def test_extract_think_trace_finds_trace_inside_candidate_text() -> None:
    text = "prefix\n<think>\nuse symmetry\n</think>\n12\n34"

    assert extract_think_trace(text) == "<think>\nuse symmetry\n</think>"


def test_completion_from_candidate_can_keep_trace_then_emit_clean_grid() -> None:
    row = candidate_row(
        0,
        [[1, 2], [3, 4]],
        exact=True,
        candidate_text="Here is one answer.\n<think>\nuse symmetry\n</think>\nwrong looking clutter",
    )

    assert (
        completion_from_candidate(row, completion_source="trace_then_canonical_grid", output_format="compact")
        == "<think>\nuse symmetry\n</think>\n12\n34"
    )


def test_completion_from_candidate_trace_then_canonical_falls_back_to_grid_without_trace() -> None:
    row = candidate_row(0, [[1, 2]], exact=True, candidate_text="The answer is 12.")

    assert completion_from_candidate(row, completion_source="trace_then_canonical_grid", output_format="compact") == "12"


def test_build_distill_rows_groups_candidates_and_writes_metadata() -> None:
    rows = [
        candidate_row(0, [[9]], exact=False, selected=True),
        candidate_row(1, [[6]], exact=True, source="symbolic_copy", candidate_text="<think>\ncopy\n</think>\n6"),
        {
            **candidate_row(0, [[4]], exact=True, selected=True, source="model_tta_identity", candidate_text="4"),
            "task_id": "task2",
            "selector_generated": True,
        },
    ]

    output_rows = build_distill_rows(
        FakeTokenizer(),
        rows,
        choice="best_exact",
        completion_source="trace_then_canonical_grid",
        output_format="compact",
        append_eos=True,
        source_jsonl="candidates.jsonl",
    )

    assert len(output_rows) == 2
    assert output_rows[0]["task_id"] == "task"
    assert output_rows[0]["candidate_index"] == 1
    assert str(output_rows[0]["completion"]).endswith("<eos>")
    assert str(output_rows[0]["completion"]).startswith("<think>\ncopy\n</think>\n6")
    assert output_rows[0]["cot"] == "<think>\ncopy\n</think>"
    assert output_rows[0]["completion_source"] == "trace_then_canonical_grid"
    assert output_rows[0]["source_dataset"] == "arc-agi-candidate-distill"
    assert output_rows[0]["selector_generated"] is False
    assert output_rows[1]["task_id"] == "task2"
    assert output_rows[1]["selector_generated"] is True
    assert output_rows[1]["selected"] is True
