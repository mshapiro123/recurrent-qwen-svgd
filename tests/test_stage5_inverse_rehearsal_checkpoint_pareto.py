from __future__ import annotations

from pathlib import Path

from colab.run_stage5_inverse_rehearsal_checkpoint_pareto import (
    SYNTHETIC_FLOOR,
    assess_checkpoint_pareto,
    parse_steps,
)


def test_pareto_assessment_requires_all_existing_gates() -> None:
    assessment = assess_checkpoint_pareto(
        task_correct=46,
        task_total=64,
        synthetic_min=SYNTHETIC_FLOOR,
        natural_accuracy=0.85671876,
        natural_baseline_accuracy=0.88671875,
    )

    assert assessment["all_current_gates_passed"] is True
    assert assessment["selection_status"] == "candidate_requires_fresh_confirmation"


def test_pareto_assessment_keeps_natural_hard_stop_disqualifying() -> None:
    assessment = assess_checkpoint_pareto(
        task_correct=64,
        task_total=64,
        synthetic_min=1.0,
        natural_accuracy=0.8567,
        natural_baseline_accuracy=0.88671875,
    )

    assert assessment["natural"]["passed"] is False
    assert assessment["all_current_gates_passed"] is False


def test_parse_steps_defaults_and_rejects_non_positive_values() -> None:
    assert parse_steps(None) == (100, 200, 300, 334)
    assert parse_steps("25, 50") == (25, 50)
    try:
        parse_steps("0,100")
    except ValueError:
        pass
    else:
        raise AssertionError("Non-positive checkpoint step was accepted")


def test_checkpoint_pareto_target_is_wired() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_INVERSE_REHEARSAL_CHECKPOINT_PARETO_CELL.py").read_text(encoding="utf-8")

    assert '"inverse_rehearsal_checkpoint_pareto"' in bootstrap
    assert "STAGE5_INVERSE_REHEARSAL_CHECKPOINT_PARETO_CELL_VERSION" in cell
    assert "candidate_requires_fresh_confirmation" in cell
    assert "STAGE5_REHEARSAL_PARETO_DISCONNECT" in cell
