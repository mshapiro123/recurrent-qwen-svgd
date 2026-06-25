import json
from pathlib import Path

from colab.review_stage5_competence_pipeline import build_review, report_lines


def test_competence_review_routes_stale_checkpoint_restore_failure(tmp_path: Path) -> None:
    source = tmp_path / "competence" / "summary.json"
    source.parent.mkdir()
    (source.parent / "arc_mix.log").write_text(
        "FileNotFoundError: Missing recovered checkpoint outputs/stage5/recovery/phase1.pt. "
        "Could not restore it from Drive.",
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(
            {
                "kind": "stage5_competence_preserving_pipeline",
                "run_id": "competence_run",
                "source_summary": "outputs/stage5/source/summary.json",
                "status": "pipeline_failed",
                "failed_stage": "arc_mix",
                "arc_mix_run_id": "competence_run_arc_mix",
                "full_assessment_run_id": "competence_run_full_assessment",
            }
        ),
        encoding="utf-8",
    )

    review = build_review(source)

    assert review["status"] == "pipeline_failed"
    assert review["next_action"]["name"] == "Resume competence-preserving recurrent recovery pipeline"
    assert "run_stage5_competence_preserving_pipeline.py" in review["next_action"]["command"]
    assert "STAGE5_COMPETENCE_ARC_MIX_RUN_ID=competence_run_arc_mix" in review["next_action"]["command"]
    assert "Could not restore it from Drive" not in "\n".join(report_lines(review))


def test_competence_review_routes_full_assessment_pass_to_broader_benchmark(tmp_path: Path) -> None:
    source = tmp_path / "competence" / "summary.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "kind": "stage5_competence_preserving_pipeline",
                "run_id": "competence_run",
                "status": "full_assessment_balanced_nonnegative",
                "full_assessment_summary": "outputs/stage5/full/summary.json",
                "full_assessment": {
                    "kind": "stage5_recovery_full_assessment",
                    "status": "balanced_nonnegative",
                    "balanced_assessment": {
                        "status": "balanced_nonnegative",
                        "best_checkpoint": {"checkpoint": "outputs/stage5/full/phase1_step_100.pt"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    review = build_review(source)

    assert review["full_assessment_status"] == "balanced_nonnegative"
    assert review["full_assessment_best_checkpoint"] == "outputs/stage5/full/phase1_step_100.pt"
    assert review["next_action"]["name"] == "Run broader benchmark suite for balanced checkpoint"
    assert "run_stage5_benchmark_suite.py" in review["next_action"]["command"]
