from __future__ import annotations

from eval.arc_agi_program import execute_arc_program, parse_arc_program_from_text, parse_color_map_literal
from eval.arc_agi_utils import ArcAgiExample, ArcPair
from eval.arc_agi_symbolic import exact_symbolic_candidate, format_symbolic_program_trace
from eval.eval_arc_agi import evaluate_example


def test_parse_color_map_literal_accepts_int_key_pairs() -> None:
    assert parse_color_map_literal("1: 2, 0: 3") == {1: 2, 0: 3}
    assert parse_color_map_literal("'1': 2, \"0\": 3") == {1: 2, 0: 3}
    assert parse_color_map_literal("1 -> 2") is None


def test_execute_arc_program_runs_transform_and_recolor() -> None:
    example = ArcAgiExample(
        task_id="program",
        test_index=0,
        train=(
            ArcPair(input=[[1, 0], [0, 0]], output=[[2, 3], [2, 2]]),
            ArcPair(input=[[0, 4], [0, 0]], output=[[2, 2], [2, 5]]),
        ),
        test_input=[[0, 0], [6, 0]],
        test_output=[[6, 2], [2, 2]],
    )
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    assert execute_arc_program(example, "\n".join(candidate.program)) == [[6, 2], [2, 2]]


def test_execute_arc_program_runs_crop_non_background() -> None:
    example = ArcAgiExample(
        task_id="crop",
        test_index=0,
        train=(
            ArcPair(input=[[0, 0, 0], [0, 4, 5], [0, 6, 0]], output=[[4, 5], [6, 0]]),
            ArcPair(input=[[0, 7, 0], [0, 8, 0], [0, 0, 0]], output=[[7], [8]]),
        ),
        test_input=[[0, 0, 0, 0], [0, 2, 3, 0], [0, 0, 4, 0]],
        test_output=[[2, 3], [0, 4]],
    )
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    assert execute_arc_program(example, "\n".join(candidate.program)) == [[2, 3], [0, 4]]


def test_execute_arc_program_runs_crop_recolor() -> None:
    example = ArcAgiExample(
        task_id="crop-recolor",
        test_index=0,
        train=(
            ArcPair(
                input=[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 1, 0], [0, 0, 0, 0]],
                output=[[6, 7], [8, 6]],
            ),
            ArcPair(input=[[0, 0, 2, 0], [0, 0, 1, 0], [0, 0, 0, 0]], output=[[7], [6]]),
        ),
        test_input=[[0, 0, 0, 0], [0, 3, 2, 0], [0, 1, 0, 0]],
        test_output=[[8, 7], [6, 0]],
    )
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    assert execute_arc_program(example, "\n".join(candidate.program)) == [[8, 7], [6, 0]]


def test_parse_arc_program_from_text_extracts_think_region() -> None:
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
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    text = "Reasoning:\n" + format_symbolic_program_trace(candidate)
    assert parse_arc_program_from_text(example, text) == [[9, 9]]


def test_evaluate_example_falls_back_to_program_execution() -> None:
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
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    rows, summary = evaluate_example(
        example,
        [format_symbolic_program_trace(candidate)],
        candidate_sources=["model"],
        diagnostics={},
        generation_steps=1,
        output_format="compact",
    )
    assert summary["best_of_k_exact"] is True
    assert rows[0]["parse_method"] == "program"
    assert rows[0]["parsed_grid"] == [[9, 9]]


def test_evaluate_example_can_prefer_program_over_wrong_literal_grid() -> None:
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
    candidate = exact_symbolic_candidate(example)
    assert candidate is not None
    text = "00\n" + format_symbolic_program_trace(candidate)

    fallback_rows, fallback_summary = evaluate_example(
        example,
        [text],
        candidate_sources=["model"],
        diagnostics={},
        generation_steps=1,
        output_format="compact",
        program_parse_mode="fallback",
    )
    assert fallback_summary["best_of_k_exact"] is False
    assert fallback_rows[0]["parse_method"] == "grid"

    prefer_rows, prefer_summary = evaluate_example(
        example,
        [text],
        candidate_sources=["model"],
        diagnostics={},
        generation_steps=1,
        output_format="compact",
        program_parse_mode="prefer",
    )
    assert prefer_summary["best_of_k_exact"] is True
    assert prefer_rows[0]["parse_method"] == "program"
