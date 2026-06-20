from __future__ import annotations

from eval.arc_agi_utils import ArcAgiExample, ArcPair
from eval.arc_agi_symbolic import exact_symbolic_candidate
from eval.eval_arc_agi import evaluate_example, format_symbolic_candidate_texts, summarize_program_verifier


def _constant_example() -> ArcAgiExample:
    return ArcAgiExample(
        task_id="constant",
        test_index=0,
        train=(
            ArcPair(input=[[1]], output=[[9, 9]]),
            ArcPair(input=[[2]], output=[[9, 9]]),
        ),
        test_input=[[3]],
        test_output=[[9, 9]],
    )


def test_format_symbolic_candidate_texts_can_emit_grid_program_or_both() -> None:
    example = _constant_example()
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None

    grid_rows = format_symbolic_candidate_texts([candidate], output_format="compact", candidate_format="grid")
    program_rows = format_symbolic_candidate_texts([candidate], output_format="compact", candidate_format="program")
    both_rows = format_symbolic_candidate_texts([candidate], output_format="compact", candidate_format="both")

    assert [source for _text, source in grid_rows] == ["symbolic_grid"]
    assert [source for _text, source in program_rows] == ["symbolic_program"]
    assert [source for _text, source in both_rows] == ["symbolic_grid", "symbolic_program"]
    assert "program:" in program_rows[0][0]


def test_symbolic_program_candidate_is_verified_and_selectable() -> None:
    example = _constant_example()
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    rows_with_sources = format_symbolic_candidate_texts(
        [candidate],
        output_format="compact",
        candidate_format="program",
    )

    rows, summary = evaluate_example(
        example,
        [text for text, _source in rows_with_sources],
        candidate_sources=[source for _text, source in rows_with_sources],
        diagnostics={},
        generation_steps=0,
        output_format="compact",
        program_parse_mode="prefer",
    )
    verifier = summarize_program_verifier(rows)

    assert summary["selected_exact"] is True
    assert rows[0]["candidate_source"] == "symbolic_program"
    assert rows[0]["parse_method"] == "program"
    assert rows[0]["program_fits_train"] is True
    assert verifier["candidates_with_program"] == 1
    assert verifier["candidates_program_fits_train"] == 1
    assert verifier["program_fit_selected_exact"] == 1
