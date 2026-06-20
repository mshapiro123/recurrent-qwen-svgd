from __future__ import annotations

from eval.analyze_arc_agi_symbolic import analyze_examples
from eval.arc_agi_utils import ArcAgiExample, ArcPair


def test_analyze_examples_reports_exact_symbolic_coverage() -> None:
    covered = ArcAgiExample(
        task_id="covered",
        test_index=0,
        train=(ArcPair(input=[[1]], output=[[9]]), ArcPair(input=[[2]], output=[[9]])),
        test_input=[[3]],
        test_output=[[9]],
    )
    uncovered = ArcAgiExample(
        task_id="uncovered",
        test_index=0,
        train=(ArcPair(input=[[1]], output=[[2]]), ArcPair(input=[[2]], output=[[4]])),
        test_input=[[3]],
        test_output=[[9]],
    )
    payload = analyze_examples([covered, uncovered])
    summary = payload["summary"]
    assert summary["examples_with_targets"] == 2
    assert summary["exact_symbolic"] == 1
    assert summary["exact_by_source"]["constant_output"] == 1
    assert summary["tasks_solved_symbolic"] == 1
