from __future__ import annotations

from eval.arc_agi_utils import (
    ArcAgiExample,
    ArcPair,
    apply_geometry_transform,
    format_grid_completion,
    parse_grid_from_text,
    transform_arc_example,
)
from eval.arc_agi_symbolic import exact_symbolic_candidate
from eval.eval_arc_agi import (
    evaluate_example,
    format_symbolic_candidate_texts,
    geometry_tta_enabled,
    invert_tta_candidate_text,
    summarize_program_verifier,
    summarize_task_families,
    task_family,
)


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


def test_task_family_extracts_synthetic_family_names() -> None:
    assert task_family("synthetic_move_recolor_000123") == "move_recolor"
    assert task_family("synthetic_frame_object_000001:loo0") == "frame_object"
    assert task_family("0d3d703e") == "arc"


def test_geometry_tta_inverts_transformed_candidate_text_before_scoring() -> None:
    example = ArcAgiExample(
        task_id="geom",
        test_index=0,
        train=(ArcPair(input=[[1, 2], [3, 4]], output=[[4, 3], [2, 1]]),),
        test_input=[[5, 6], [7, 8]],
        test_output=[[8, 7], [6, 5]],
    )
    transformed = transform_arc_example(example, "rot90")
    transformed_prediction = apply_geometry_transform(example.test_output, "rot90")

    restored_text = invert_tta_candidate_text(
        transformed,
        format_grid_completion(transformed_prediction, output_format="compact"),
        "rot90",
        output_format="compact",
        program_parse_mode="fallback",
    )

    assert geometry_tta_enabled("rot90")
    assert not geometry_tta_enabled("none")
    assert parse_grid_from_text(restored_text, output_format="compact") == example.test_output


def test_summarize_task_families_tracks_family_exact_rates() -> None:
    rows = [
        {
            "task_id": "synthetic_move_recolor_000001",
            "has_target": True,
            "first_exact": True,
            "selected_exact": True,
            "best_of_k_exact": True,
            "valid_candidates": 1,
            "num_candidates": 1,
        },
        {
            "task_id": "synthetic_frame_object_000002",
            "has_target": True,
            "first_exact": False,
            "selected_exact": False,
            "best_of_k_exact": True,
            "valid_candidates": 1,
            "num_candidates": 2,
        },
    ]

    summary = summarize_task_families(rows)

    assert summary["move_recolor"]["selected_exact"] == 1
    assert summary["move_recolor"]["task_solve_rate_best_of_k"] == 1.0
    assert summary["frame_object"]["best_of_k_exact"] == 1
    assert summary["frame_object"]["valid_candidate_rate"] == 0.5
