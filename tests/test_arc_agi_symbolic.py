from __future__ import annotations

from eval.arc_agi_symbolic import learn_color_map, symbolic_candidates
from eval.arc_agi_utils import ArcAgiExample, ArcPair
from eval.eval_arc_agi import evaluate_example, summarize_candidate_sources


def test_learn_color_map_rejects_conflicts() -> None:
    assert learn_color_map([[[1], [1]]], [[[2], [3]]]) is None


def test_symbolic_candidates_learn_geometry_plus_color_map() -> None:
    example = ArcAgiExample(
        task_id="rot-color",
        test_index=0,
        train=(
            ArcPair(input=[[1, 0], [0, 0]], output=[[2, 3], [2, 2]]),
            ArcPair(input=[[0, 4], [0, 0]], output=[[2, 2], [2, 5]]),
        ),
        test_input=[[0, 0], [6, 0]],
        test_output=[[6, 2], [2, 2]],
    )
    candidates = symbolic_candidates(example)
    assert any(candidate.grid == [[6, 2], [2, 2]] for candidate in candidates)


def test_symbolic_candidates_include_constant_output() -> None:
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
    assert symbolic_candidates(example)[0].name == "constant_output"
    assert symbolic_candidates(example)[0].grid == [[9, 9]]


def test_evaluate_example_tracks_candidate_sources() -> None:
    example = ArcAgiExample(
        task_id="source",
        test_index=0,
        train=(ArcPair(input=[[1]], output=[[2]]),),
        test_input=[[3]],
        test_output=[[4]],
    )
    rows, summary = evaluate_example(
        example,
        ["0", "4"],
        candidate_sources=["model", "symbolic"],
        diagnostics={},
        generation_steps=1,
        output_format="compact",
    )
    assert summary["best_of_k_exact"] is True
    by_source = summarize_candidate_sources(rows)
    assert by_source["symbolic"]["exact"] == 1
