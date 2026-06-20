from __future__ import annotations

from eval.analyze_arc_agi_recovery import analyze_recovery, classify_example, family_gap_rows


def _example(
    task_id: str,
    *,
    selected: bool,
    best: bool,
    valid: int = 1,
    candidates: int = 1,
    test_index: int = 0,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "test_index": test_index,
        "has_target": True,
        "first_exact": selected,
        "selected_exact": selected,
        "best_of_k_exact": best,
        "valid_candidates": valid,
        "num_candidates": candidates,
        "selected_index": 0,
    }


def _payload(*examples: dict[str, object]) -> dict[str, object]:
    return {
        "summary": {
            "examples_with_targets": len(examples),
            "selected_exact": sum(1 for item in examples if item["selected_exact"]),
            "best_of_k_exact": sum(1 for item in examples if item["best_of_k_exact"]),
        },
        "examples": list(examples),
        "candidate_source_summary": {},
        "parse_method_summary": {},
        "program_verifier_summary": {},
    }


def test_classify_example_separates_selector_and_generation_failures() -> None:
    assert classify_example(_example("ok", selected=True, best=True)) == "selected_exact"
    assert classify_example(_example("miss", selected=False, best=True)) == "selector_miss"
    assert classify_example(_example("invalid", selected=False, best=False, valid=0)) == "no_valid_candidate"
    assert classify_example(_example("wrong", selected=False, best=False, valid=1)) == "no_exact_candidate"


def test_family_gap_rows_rank_base_only_regressions() -> None:
    base = _payload(
        _example("synthetic_move_recolor_000001", selected=True, best=True),
        _example("synthetic_frame_object_000001", selected=True, best=True),
    )
    recovered = _payload(
        _example("synthetic_move_recolor_000001", selected=False, best=False, valid=1),
        _example("synthetic_frame_object_000001", selected=True, best=True),
    )

    rows = family_gap_rows(base, recovered, reference_label="base", candidate_label="recovered")

    assert rows[0]["family"] == "move_recolor"
    assert rows[0]["selected_delta"] == -1
    assert rows[0]["best_of_k_delta"] == -1
    assert rows[0]["recovered_no_exact_candidate"] == 1


def test_analyze_recovery_recommends_generation_and_selector_work() -> None:
    base = _payload(
        _example("synthetic_move_recolor_000001", selected=True, best=True),
        _example("synthetic_frame_object_000001", selected=True, best=True),
        _example("synthetic_crop_000001", selected=False, best=False),
    )
    start = _payload(
        _example("synthetic_move_recolor_000001", selected=False, best=False),
        _example("synthetic_frame_object_000001", selected=False, best=False),
        _example("synthetic_crop_000001", selected=False, best=False),
    )
    recovered = _payload(
        _example("synthetic_move_recolor_000001", selected=False, best=False, valid=1),
        _example("synthetic_frame_object_000001", selected=False, best=True, valid=1),
        _example("synthetic_crop_000001", selected=False, best=False, valid=0),
    )

    analysis = analyze_recovery(base=base, start=start, recovered=recovered)
    areas = [item["area"] for item in analysis["recommendations"]]

    assert "curriculum_generation" in areas
    assert "selector_or_tta" in areas
    assert "format_parse" in areas
    assert "scale_recovery" in areas
    assert analysis["models"]["recovered"]["failure_buckets"]["selector_miss"] == 1
    assert analysis["regression_examples"]["base_selected_recovered_missed"][0]["family"] == "frame_object"
