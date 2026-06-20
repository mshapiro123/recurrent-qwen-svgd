from __future__ import annotations

import json

from eval.arc_agi_utils import (
    load_arc_agi_examples,
    parse_grid_from_text,
    render_arc_prompt,
    score_grid_prediction,
    validate_grid,
)


def test_validate_grid_rejects_ragged_rows() -> None:
    try:
        validate_grid([[1, 2], [3]])
    except ValueError as exc:
        assert "rectangular" in str(exc)
    else:
        raise AssertionError("ragged grid should fail validation")


def test_parse_grid_from_text_extracts_first_valid_grid() -> None:
    text = 'Reasoning...\n```json\n[[1,2,3],[4,5,6]]\n```\nDone'
    assert parse_grid_from_text(text) == [[1, 2, 3], [4, 5, 6]]


def test_parse_grid_from_text_skips_invalid_arrays() -> None:
    text = "bad [1,2,3] then good [[0],[9]]"
    assert parse_grid_from_text(text) == [[0], [9]]


def test_load_arc_agi_directory_and_render_prompt(tmp_path) -> None:
    task = {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
            {"input": [[3]], "output": [[4]]},
        ],
        "test": [{"input": [[5]], "output": [[6]]}],
    }
    (tmp_path / "abc123.json").write_text(json.dumps(task), encoding="utf-8")

    examples = load_arc_agi_examples(tmp_path)
    assert len(examples) == 1
    assert examples[0].task_id == "abc123"
    assert examples[0].test_output == [[6]]
    prompt = render_arc_prompt(examples[0])
    assert "Training example 1 input:" in prompt
    assert "Output JSON grid:" in prompt


def test_load_arc_agi_combined_challenges_with_solutions(tmp_path) -> None:
    challenges = {
        "task-a": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}, {"input": [[4]]}],
        }
    }
    solutions = {"task-a": [[[5]], [[6]]]}
    challenge_path = tmp_path / "challenges.json"
    solution_path = tmp_path / "solutions.json"
    challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
    solution_path.write_text(json.dumps(solutions), encoding="utf-8")

    examples = load_arc_agi_examples(challenge_path, solutions_path=solution_path)
    assert [(item.test_index, item.test_output) for item in examples] == [(0, [[5]]), (1, [[6]])]


def test_score_grid_prediction() -> None:
    assert score_grid_prediction([[1]], [[1]])["exact"] is True
    assert score_grid_prediction([[1, 2]], [[1], [2]])["shape_match"] is False
    assert score_grid_prediction(None, [[1]])["valid"] is False
    assert score_grid_prediction([[1]], None)["has_target"] is False
