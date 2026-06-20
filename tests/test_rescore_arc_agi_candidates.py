from __future__ import annotations

from eval.rescore_arc_agi_candidates import (
    build_payload,
    read_inferred_shapes,
    rescore_groups,
)


def candidate_row(
    idx: int,
    grid: list[list[int]] | None,
    *,
    exact: bool,
    source: str = "model",
    selected: bool = False,
    parse_method: str = "grid",
    program_fits_train: bool = False,
) -> dict:
    return {
        "task_id": "task",
        "test_index": 0,
        "candidate_index": idx,
        "candidate_source": source,
        "candidate_text": str(grid),
        "parsed_grid": grid,
        "parse_method": parse_method,
        "program_train_matches": 0,
        "program_train_total": 1 if parse_method == "program" else 0,
        "program_fits_train": program_fits_train,
        "target_grid": [[6, 6]],
        "score": {"valid": grid is not None, "exact": exact},
        "selected": selected,
    }


def test_read_inferred_shapes_from_summary_payload() -> None:
    payload = {
        "examples": [
            {
                "task_id": "task",
                "test_index": 0,
                "inferred_shapes": [[1, 2], ["bad"], [0, 3], [1, 2]],
            }
        ]
    }

    assert read_inferred_shapes(payload) == {("task", 0): [(1, 2)]}


def test_rescore_self_consistency_replays_tta_votes_offline() -> None:
    rows = [
        candidate_row(0, [[9, 9]], exact=False, source="model_tta_identity", selected=True),
        candidate_row(1, [[6, 6]], exact=True, source="model_tta_rot90"),
        candidate_row(2, [[6, 6]], exact=True, source="model_tta_rot180"),
        candidate_row(3, [[0], [0]], exact=False, source="model_tta_flip_h"),
        candidate_row(4, [[0], [0]], exact=False, source="model_tta_flip_v"),
    ]

    rescored, summaries = rescore_groups(
        rows,
        inferred_shapes_by_key={("task", 0): [(1, 2)]},
        selection_strategy="self_consistency",
    )

    selected = [row for row in rescored if row["selected"]]
    assert selected[0]["candidate_index"] == 1
    assert selected[0]["score"]["exact"] is True
    assert rescored[0]["previous_selected"] is True
    assert summaries[0]["selected_exact"] is True
    assert summaries[0]["best_of_k_exact"] is True
    assert summaries[0]["valid_candidates"] == 5


def test_rescore_symbolic_priority_selects_later_symbolic_candidate() -> None:
    rows = [
        candidate_row(0, [[9, 9]], exact=False, source="model", selected=True),
        candidate_row(1, [[6, 6]], exact=True, source="symbolic_copy"),
    ]

    rescored, summaries = rescore_groups(
        rows,
        inferred_shapes_by_key={("task", 0): [(1, 2)]},
        selection_strategy="symbolic_priority",
    )

    selected = [row for row in rescored if row["selected"]]
    assert selected[0]["candidate_index"] == 1
    assert summaries[0]["selected_index"] == 1
    assert summaries[0]["selected_exact"] is True


def test_rescore_verified_program_still_beats_symbolic_priority() -> None:
    rows = [
        candidate_row(0, [[9, 9]], exact=False, source="model", selected=True),
        candidate_row(1, [[6, 6]], exact=True, source="symbolic_copy"),
        candidate_row(
            2,
            [[1, 1]],
            exact=False,
            source="model_program",
            parse_method="program",
            program_fits_train=True,
        ),
    ]

    rescored, summaries = rescore_groups(
        rows,
        inferred_shapes_by_key={("task", 0): [(1, 2)]},
        selection_strategy="symbolic_priority",
    )

    selected = [row for row in rescored if row["selected"]]
    assert selected[0]["candidate_index"] == 2
    assert summaries[0]["selected_exact"] is False


def test_build_payload_recomputes_selected_summary() -> None:
    rows = [
        candidate_row(0, [[9, 9]], exact=False, source="model", selected=True),
        candidate_row(1, [[6, 6]], exact=True, source="symbolic_copy"),
    ]
    rescored, summaries = rescore_groups(
        rows,
        inferred_shapes_by_key={("task", 0): [(1, 2)]},
        selection_strategy="symbolic_priority",
    )

    payload = build_payload(
        candidates_jsonl="candidates.jsonl",
        summary_payload={"tasks_path": "tasks.jsonl", "grid_format": "json", "summary": {"selected_exact": 0}},
        rescored_rows=rescored,
        example_summaries=summaries,
        selection_strategy="symbolic_priority",
    )

    assert payload["selection_strategy"] == "symbolic_priority"
    assert payload["summary"]["selected_exact"] == 1
    assert payload["candidate_source_summary"]["symbolic_copy"]["selected_exact"] == 1
