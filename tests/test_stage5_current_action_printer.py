from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_action_printer_names_score_alignment_target() -> None:
    script = (ROOT / "colab/print_current_stage5_action.py").read_text(encoding="utf-8")

    assert 'PREFERRED_TARGET = "traced_sft_score_alignment_repair"' in script
    assert "colab/plan_stage5_next_run.py" in script
    assert "stage5_current_source_summary.txt" in script
